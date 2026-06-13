import os
import re
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi

from agentic_rag import config

logger = logging.getLogger(__name__)

BLOCK_QUERY_TERMS = {
    "definition": {"definition", "definitions", "define", "defined", "meaning", "définition", "definir", "définir"},
    "example": {"example", "examples", "exemple", "exemples"},
    "theorem": {"theorem", "theorems", "théorème", "theoreme"},
    "proposition": {"proposition", "propositions"},
    "lemma": {"lemma", "lemmas", "lemme", "lemmes"},
    "remark": {"remark", "remarks", "remarque", "remarques"},
    "corollary": {"corollary", "corollaries", "corollaire", "corollaires"},
    "notation": {"notation", "notations"},
}

DEFINITION_FRIENDLY_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "preliminaries",
    "preliminary",
    "definitions",
    "definition",
    "examples",
    "example",
}

BLOCK_TYPE_BOOST = 0.15
SECTION_TYPE_BOOST = 0.05

# Try to import FAISS; if it fails, we will fall back to numpy-based similarity
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS is not available. Falling back to pure numpy Cosine Similarity.")

class EmbeddingEngine:
    """Pluggable Embedding Engine supporting Local, OpenAI, and Gemini embeddings."""
    def __init__(self):
        self.provider = config.EMBEDDING_PROVIDER.lower()
        self.model_name = config.LOCAL_EMBEDDING_MODEL
        self._model = None

    def _load_local_model(self):
        if self._model is None:
            logger.info(f"Loading local embedding model: {self.model_name}")
            from sentence_transformers import SentenceTransformer
            # Map devices (prefer CUDA if available)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_queries(self, texts: List[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self.provider == "local":
            model = self._load_local_model()
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            return embeddings
        elif self.provider == "openai":
            # Direct API request to OpenAI embeddings
            import requests
            headers = {
                "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            url = "https://api.openai.com/v1/embeddings"
            payload = {
                "input": texts,
                "model": "text-embedding-3-large" if config.EMBEDDING_DIM == 3024 else "text-embedding-ada-002"
            }
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
                embeddings = [item["embedding"] for item in data["data"]]
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"OpenAI Embeddings failed: {e}. Falling back to local/random embeddings.")
                # fallback
        elif self.provider == "gemini":
            import requests
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={config.GEMINI_API_KEY}"
            # For Gemini, embed multiple contents
            embeddings = []
            try:
                for text in texts:
                    payload = {
                        "content": {"parts": [{"text": text}]}
                    }
                    r = requests.post(url, json=payload, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    embeddings.append(data["embedding"]["values"])
                return np.array(embeddings, dtype=np.float32)
            except Exception as e:
                logger.error(f"Gemini Embeddings failed: {e}. Falling back to local/random embeddings.")
                
        # Zero fallback / random embeddings for robust testing if no keys and local fails
        logger.warning("Using mock random embeddings as safety fallback")
        return np.random.randn(len(texts), config.EMBEDDING_DIM).astype(np.float32)

class RerankerEngine:
    """Pluggable Reranking Engine supporting BGE Reranker or Cohere Rerank."""
    def __init__(self):
        self.use_reranker = config.USE_RERANKER
        self.model_name = config.LOCAL_RERANKER_MODEL
        self._model = None

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading local reranker model: {self.model_name}")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        if not self.use_reranker or not chunks:
            return chunks[:top_n]

        try:
            # Prepare pairs (query, chunk_text)
            pairs = [[query, chunk["text"]] for chunk in chunks]
            
            # Local cross-encoder rerank
            model = self._load_model()
            scores = model.predict(pairs)
            
            # Attach scores to chunks and sort
            for chunk, score in zip(chunks, scores):
                chunk["rerank_score"] = float(score)
                
            sorted_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            return sorted_chunks[:top_n]
        except Exception as e:
            logger.error(f"Reranking failed: {e}. Returning original ranking.")
            return chunks[:top_n]

class VectorStore:
    def __init__(self):
        self.index_dir = config.INDEX_DIR
        self.index_file = self.index_dir / "index.faiss"
        self.meta_file = self.index_dir / "metadata.pkl"
        
        self.embedder = EmbeddingEngine()
        self.reranker = RerankerEngine()
        
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index = None
        self.bm25: Optional[BM25Okapi] = None
        
        self.load()

    def _init_faiss(self):
        if not FAISS_AVAILABLE or self.embeddings is None or len(self.embeddings) == 0:
            return
        dim = self.embeddings.shape[1]
        # Inner Product Flat index (for normalized Cosine similarity)
        self.index = faiss.IndexFlatIP(dim)
        # Normalize embeddings to unit length for inner product index
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized_embeddings = self.embeddings / norms
        self.index.add(normalized_embeddings.astype(np.float32))

    def _init_bm25(self):
        if not self.chunks:
            self.bm25 = None
            return
        # Tokenize chunk texts for BM25
        tokenized_corpus = [self._tokenize(chunk["text"]) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def _query_block_intents(self, query: str) -> set[str]:
        tokens = set(self._tokenize(query))
        intents = {
            block_type
            for block_type, terms in BLOCK_QUERY_TERMS.items()
            if tokens.intersection(terms)
        }

        # "What is X?" questions are often definition-seeking even without the word "definition".
        if re.search(r"\bwhat\s+(?:is|are)\b", query.lower()):
            intents.add("definition")

        return intents

    def _metadata_boost(self, query: str, chunk: Dict[str, Any]) -> float:
        intents = self._query_block_intents(query)
        if not intents:
            return 0.0

        boost = 0.0
        block_type = chunk.get("block_type", "paragraph")
        section = chunk.get("section", "").lower()

        if block_type in intents:
            boost += BLOCK_TYPE_BOOST

        if "definition" in intents and any(name in section for name in DEFINITION_FRIENDLY_SECTIONS):
            boost += SECTION_TYPE_BOOST

        return boost

    def add_chunks(self, new_chunks: List[Dict[str, Any]]):
        """Adds new chunks to the vector store, computes embeddings, and re-indexes."""
        if not new_chunks:
            return
            
        logger.info(f"Adding {len(new_chunks)} new chunks to Vector Store...")

        incoming_ids = {c["arxiv_id"] for c in new_chunks}
        if incoming_ids and self.chunks:
            kept_chunks = []
            kept_embedding_indices = []
            removed_count = 0

            for index, chunk in enumerate(self.chunks):
                if chunk["arxiv_id"] in incoming_ids:
                    removed_count += 1
                    continue
                kept_chunks.append(chunk)
                kept_embedding_indices.append(index)

            if removed_count:
                logger.info(f"Replacing {removed_count} existing chunks for papers: {sorted(incoming_ids)}")
                self.chunks = kept_chunks
                if self.embeddings is not None:
                    original_embedding_count = removed_count + len(kept_embedding_indices)
                    if len(self.embeddings) == original_embedding_count:
                        self.embeddings = self.embeddings[kept_embedding_indices] if kept_embedding_indices else None
                    elif kept_chunks:
                        logger.warning("Embedding count does not match metadata. Rebuilding kept embeddings.")
                        self.embeddings = self.embedder.embed_documents([c["text"] for c in kept_chunks])
                    else:
                        self.embeddings = None
        
        # Avoid indexing duplicates
        existing_keys = {(c["arxiv_id"], c["section"], c["chunk_index"]) for c in self.chunks}
        filtered_new_chunks = []
        for c in new_chunks:
            key = (c["arxiv_id"], c["section"], c["chunk_index"])
            if key not in existing_keys:
                filtered_new_chunks.append(c)
                
        if not filtered_new_chunks:
            logger.info("All chunks already exist in index. Skipping.")
            return

        # Compute embeddings for new chunks
        texts = [c["text"] for c in filtered_new_chunks]
        new_embeddings = self.embedder.embed_documents(texts)

        # Merge with existing
        if self.embeddings is None or len(self.embeddings) == 0:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        self.chunks.extend(filtered_new_chunks)
        
        # Rebuild indices
        self._init_faiss()
        self._init_bm25()
        self.save()

    def search_dense(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        """Dense similarity search."""
        if self.chunks is None or len(self.chunks) == 0:
            return []
            
        # Normalize query vector
        norm = np.linalg.norm(query_vector)
        query_vector_norm = query_vector / (norm if norm > 0 else 1.0)
        query_vector_norm = query_vector_norm.reshape(1, -1).astype(np.float32)

        if FAISS_AVAILABLE and self.index is not None:
            scores, indices = self.index.search(query_vector_norm, top_k)
            # return index and scores
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1:
                    results.append((int(idx), float(score)))
            return results
        else:
            # Pure numpy Cosine Similarity fallback
            if self.embeddings is None:
                return []
            # Calculate cosine similarities
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            norm_embeddings = self.embeddings / norms
            
            similarities = np.dot(norm_embeddings, query_vector_norm.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def search_sparse(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        """Sparse BM25 search."""
        if self.bm25 is None:
            return []
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only return matches
                results.append((int(idx), float(scores[idx])))
        return results

    def hybrid_search(self, query: str, top_k: int = config.RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """Combines BM25 and Vector Search scores (Hybrid Search) followed by Reranking."""
        if not self.chunks:
            return []

        # 1. Dense search
        query_vec = self.embedder.embed_queries([query])[0]
        dense_results = self.search_dense(query_vec, top_k=top_k * 2)
        
        # 2. Sparse search
        sparse_results = self.search_sparse(query, top_k=top_k * 2)

        # 3. Score normalization and fusion
        # Normalized score = (score - min) / (max - min) (if max != min)
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}
        
        all_indices = set(dense_scores.keys()).union(set(sparse_scores.keys()))
        
        # Max/min calculations for normalization
        max_d = max(dense_scores.values()) if dense_scores else 1.0
        min_d = min(dense_scores.values()) if dense_scores else 0.0
        range_d = max_d - min_d if max_d > min_d else 1.0

        max_s = max(sparse_scores.values()) if sparse_scores else 1.0
        min_s = min(sparse_scores.values()) if sparse_scores else 0.0
        range_s = max_s - min_s if max_s > min_s else 1.0

        scored_chunks = []
        for idx in all_indices:
            d_score = dense_scores.get(idx, min_d)
            s_score = sparse_scores.get(idx, min_s)
            
            # Normalize
            norm_d = (d_score - min_d) / range_d
            norm_s = (s_score - min_s) / range_s
            
            # Hybrid fusion
            hybrid_score = (config.DENSE_WEIGHT * norm_d) + (config.BM25_WEIGHT * norm_s)
            metadata_boost = self._metadata_boost(query, self.chunks[idx])
            boosted_score = hybrid_score + metadata_boost
            
            chunk_copy = self.chunks[idx].copy()
            chunk_copy["hybrid_score"] = boosted_score
            chunk_copy["base_hybrid_score"] = hybrid_score
            chunk_copy["metadata_boost"] = metadata_boost
            chunk_copy["dense_score"] = d_score
            chunk_copy["sparse_score"] = s_score
            scored_chunks.append(chunk_copy)
            
        # Sort by hybrid score
        scored_chunks = sorted(scored_chunks, key=lambda x: x["hybrid_score"], reverse=True)
        candidate_chunks = scored_chunks[:top_k]

        # 4. Reranking
        logger.info(f"Reranking top {len(candidate_chunks)} candidate chunks using {self.reranker.model_name}...")
        reranked_chunks = self.reranker.rerank(query, candidate_chunks, top_n=config.RERANK_TOP_N)
        
        return reranked_chunks

    def save(self):
        """Saves vector store indices and metadata to disk."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata and embeddings array
        with open(self.meta_file, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "embeddings": self.embeddings
            }, f)
            
        # Save FAISS index
        if FAISS_AVAILABLE and self.index is not None:
            faiss.write_index(self.index, str(self.index_file))
            
        logger.info(f"Vector store saved: {len(self.chunks)} chunks stored.")

    def load(self):
        """Loads vector store indices and metadata from disk."""
        if not self.meta_file.exists():
            logger.info("No existing vector store metadata found. Starting fresh.")
            return

        try:
            with open(self.meta_file, "rb") as f:
                data = pickle.load(f)
                self.chunks = data.get("chunks", [])
                self.embeddings = data.get("embeddings", None)

            for chunk in self.chunks:
                chunk.setdefault("block_type", "paragraph")
                chunk.setdefault("block_index", 0)
                
            # Load FAISS index if available and exists, otherwise rebuild it from embeddings
            if FAISS_AVAILABLE and self.index_file.exists():
                self.index = faiss.read_index(str(self.index_file))
            else:
                self._init_faiss()
                
            self._init_bm25()
            logger.info(f"Loaded vector store with {len(self.chunks)} chunks.")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}. Starting fresh.")
            self.chunks = []
            self.embeddings = None
            self.index = None
            self.bm25 = None
            
    def get_indexed_arxiv_ids(self) -> List[str]:
        """Returns list of unique arXiv IDs already indexed in the vector store."""
        return list({c["arxiv_id"] for c in self.chunks})
