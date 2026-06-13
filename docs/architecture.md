# Architecture Interne

Cette page décrit le rôle des modules principaux et le flux de données dans le système.

## Vue d’Ensemble

```text
User prompt
  -> AgenticOrchestrator
  -> LLM decides action
  -> arXiv search / PDF ingest / retrieval
  -> VectorStore hybrid search
  -> LLM final synthesis
  -> optional RAG evaluation
```

## `agentic_rag/search.py`

Responsabilités :

- interroger l’API arXiv ;
- nettoyer les IDs arXiv ;
- parser les flux Atom XML ;
- extraire les métadonnées : titre, auteurs, résumé, date, catégories, lien PDF ;
- télécharger les PDF dans `data/pdfs/`.

Fonctions importantes :

- `clean_arxiv_id(...)`
- `ArxivSearchAgent.search(...)`
- `ArxivSearchAgent.fetch_by_ids(...)`
- `ArxivSearchAgent.download_pdf(...)`

## `agentic_rag/pdf_parser.py`

Responsabilités :

- lire les PDF avec PyMuPDF (`fitz`) ;
- extraire les blocs de texte avec taille de police, page et style ;
- détecter les titres de section ;
- regrouper le texte par section ;
- détecter les blocs importants (`definition`, `example`, `theorem`, etc.) ;
- découper les sections en chunks en préservant le type de bloc ;
- conserver les métadonnées nécessaires aux citations.

Sortie typique d’un chunk :

```json
{
  "arxiv_id": "2407.14477",
  "title": "Data-Centric Human Preference with Rationales for Direct Preference Alignment",
  "authors": "...",
  "section": "Introduction",
  "block_type": "definition",
  "block_index": 0,
  "page_start": 1,
  "page_end": 2,
  "chunk_index": 0,
  "text": "..."
}
```

## `agentic_rag/vector_db.py`

Responsabilités :

- calculer les embeddings ;
- créer ou charger un index FAISS ;
- créer un index BM25 ;
- effectuer la recherche dense seule (`faiss`) ;
- effectuer la recherche sparse seule (`bm25`) ;
- fusionner les scores (`hybrid`) ;
- booster légèrement les chunks dont le `block_type` ou la section correspond à l’intention de la question ;
- appliquer un reranker avec le mode `hybrid_reranker` ;
- sauvegarder `index.faiss` et `metadata.pkl`.

Flux de recherche :

```text
query
  -> retrieval mode: faiss | bm25 | hybrid | hybrid_reranker
  -> optional score normalization/fusion
  -> optional metadata boost
  -> optional reranking
  -> top chunks
```

Paramètres importants :

- `BM25_WEIGHT`
- `DENSE_WEIGHT`
- `RETRIEVAL_MODE`
- `RETRIEVAL_TOP_K`
- `RERANK_TOP_N`
- `USE_RERANKER`

## `agentic_rag/llm.py`

Responsabilités :

- fournir une interface commune `LLMClient.chat(...)` ;
- gérer Ollama, Gemini, OpenAI et Mock ;
- retourner des réponses via `LLMResponse`.

Clients disponibles :

- `OllamaClient`
- `GeminiClient`
- `OpenAIClient`
- `MockClient`

## `agentic_rag/orchestrator.py`

Responsabilités :

- piloter la boucle agentique ;
- transmettre au LLM la liste des outils disponibles ;
- parser la réponse JSON du LLM ;
- exécuter les actions ;
- ajouter les observations au contexte ;
- retourner la réponse finale.

Actions disponibles :

```text
search_arxiv
download_and_index
retrieve_context
answer_user
```

Le LLM doit répondre en JSON :

```json
{
  "thought": "Reasoning about the next step",
  "action": "retrieve_context",
  "action_input": {
    "query": "preference rationales alignment",
    "top_k": 10
  }
}
```

## `agentic_rag/evaluator.py`

Responsabilités :

- formater les chunks récupérés ;
- demander à un LLM juge de scorer la réponse ;
- produire trois métriques RAG Triad ;
- calculer un score global moyen ;
- générer un rapport Markdown.

Métriques :

- context relevance ;
- groundedness / faithfulness ;
- answer relevance.

## `agentic_rag/experiments.py`

Responsabilités :

- lire un dataset JSONL ;
- lancer une question RAG pour chaque ligne ;
- récupérer le contexte ;
- évaluer la réponse ;
- sauvegarder les résultats dans `runs/<experiment>/`.

Fichiers générés :

```text
runs/<experiment>/
  config.json
  results.jsonl
  summary.json
```

## `dashboard/`

Responsabilités :

- `app.py` : dashboard de qualité pour comparer les expériences ;
- `loaders.py` : lecture et normalisation des fichiers `runs/*` ;
- `query_ui.py` : UI simple pour poser un prompt et inspecter la réponse.
