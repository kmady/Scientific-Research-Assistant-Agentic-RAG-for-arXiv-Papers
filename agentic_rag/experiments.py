import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from agentic_rag import config
from agentic_rag.evaluator import RAGEvaluator, normalize_evaluation_backend
from agentic_rag.orchestrator import AgenticOrchestrator
from agentic_rag.vector_db import normalize_retrieval_mode

logger = logging.getLogger(__name__)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {e}") from e
    return records


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def current_rag_config() -> Dict[str, Any]:
    return {
        "llm_provider": config.LLM_PROVIDER,
        "ollama_model": config.OLLAMA_MODEL,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "local_embedding_model": config.LOCAL_EMBEDDING_MODEL,
        "embedding_dim": config.EMBEDDING_DIM,
        "use_reranker": config.USE_RERANKER,
        "local_reranker_model": config.LOCAL_RERANKER_MODEL,
        "retrieval_mode": config.RETRIEVAL_MODE,
        "bm25_weight": config.BM25_WEIGHT,
        "dense_weight": config.DENSE_WEIGHT,
        "retrieval_top_k": config.RETRIEVAL_TOP_K,
        "rerank_top_n": config.RERANK_TOP_N,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "evaluation_backend": config.EVALUATION_BACKEND,
    }


def safe_score(eval_results: Dict[str, Any], metric: str) -> float:
    value = eval_results.get(metric, {}).get("score", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize_results(records: List[Dict[str, Any]], experiment: str) -> Dict[str, Any]:
    count = len(records)
    if count == 0:
        return {
            "experiment": experiment,
            "questions": 0,
            "overall_rag_score": 0.0,
            "avg_context_relevance": 0.0,
            "avg_groundedness": 0.0,
            "avg_answer_relevance": 0.0,
            "avg_latency_seconds": 0.0,
        }

    def avg(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "experiment": experiment,
        "questions": count,
        "overall_rag_score": avg([r["evaluation"].get("overall_rag_score", 0.0) for r in records]),
        "avg_context_relevance": avg([safe_score(r["evaluation"], "context_relevance") for r in records]),
        "avg_groundedness": avg([safe_score(r["evaluation"], "groundedness") for r in records]),
        "avg_answer_relevance": avg([safe_score(r["evaluation"], "answer_relevance") for r in records]),
        "avg_latency_seconds": avg([r.get("latency_seconds", 0.0) for r in records]),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


class ExperimentRunner:
    def __init__(
        self,
        output_root: Path = Path("runs"),
        retrieval_mode: str | None = None,
        evaluation_backend: str | None = None,
    ):
        self.output_root = output_root
        self.retrieval_mode = normalize_retrieval_mode(retrieval_mode).value
        self.evaluation_backend = normalize_evaluation_backend(evaluation_backend)
        self.orchestrator = AgenticOrchestrator(retrieval_mode=self.retrieval_mode)
        self.evaluator = RAGEvaluator(backend=self.evaluation_backend)

    def run(self, dataset_path: Path, experiment: str, limit: int | None = None) -> Dict[str, Any]:
        questions = read_jsonl(dataset_path)
        if limit is not None:
            questions = questions[:limit]

        output_dir = self.output_root / experiment
        output_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot = current_rag_config()
        config_snapshot.update({
            "experiment": experiment,
            "dataset_path": str(dataset_path),
            "retrieval_mode": self.retrieval_mode,
            "evaluation_backend": self.evaluation_backend,
            "started_at": datetime.now(timezone.utc).isoformat(),
        })
        write_json(output_dir / "config.json", config_snapshot)

        records = []
        for item in questions:
            question_id = item.get("id", "")
            question = item["question"]
            logger.info("Running experiment '%s' question %s", experiment, question_id)

            started = time.perf_counter()
            rag_result = self.orchestrator.run(question)
            latency = time.perf_counter() - started

            retrieved = self.orchestrator.vector_store.search(
                question,
                top_k=config.RETRIEVAL_TOP_K,
                mode=self.retrieval_mode,
            )
            evaluation = self.evaluator.evaluate(question, retrieved, rag_result.get("answer", ""))

            record = {
                "id": question_id,
                "question": question,
                "question_type": item.get("question_type", ""),
                "paper_ids": item.get("paper_ids", []),
                "expected_topics": item.get("expected_topics", []),
                "answer": rag_result.get("answer", ""),
                "agent_success": rag_result.get("success", False),
                "agent_steps": rag_result.get("steps", []),
                "retrieved_chunks": [
                    {
                        "arxiv_id": c.get("arxiv_id"),
                        "title": c.get("title"),
                        "section": c.get("section"),
                        "block_type": c.get("block_type", "paragraph"),
                        "block_index": c.get("block_index"),
                        "page_start": c.get("page_start"),
                        "page_end": c.get("page_end"),
                        "chunk_index": c.get("chunk_index"),
                        "retrieval_mode": c.get("retrieval_mode", self.retrieval_mode),
                        "hybrid_score": c.get("hybrid_score"),
                        "base_hybrid_score": c.get("base_hybrid_score"),
                        "metadata_boost": c.get("metadata_boost"),
                        "dense_score": c.get("dense_score"),
                        "sparse_score": c.get("sparse_score"),
                        "rerank_score": c.get("rerank_score"),
                        "text": c.get("text"),
                    }
                    for c in retrieved
                ],
                "evaluation": evaluation,
                "latency_seconds": latency,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            records.append(record)

        append_jsonl(output_dir / "results.jsonl", records)
        summary = summarize_results(records, experiment)
        summary["retrieval_mode"] = self.retrieval_mode
        summary["evaluation_backend"] = self.evaluation_backend
        write_json(output_dir / "summary.json", summary)
        return summary
