import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from agentic_rag.evaluator import normalize_evaluation_backend
from agentic_rag.experiments import ExperimentRunner
from agentic_rag.vector_db import RetrievalMode, normalize_retrieval_mode

logger = logging.getLogger(__name__)


DEFAULT_BENCHMARK_MODES = [
    RetrievalMode.FAISS,
    RetrievalMode.BM25,
    RetrievalMode.HYBRID,
    RetrievalMode.HYBRID_RERANKER,
]


def parse_modes(raw_modes: str | None) -> List[str]:
    if not raw_modes:
        return [mode.value for mode in DEFAULT_BENCHMARK_MODES]

    modes = []
    for raw_mode in raw_modes.split(","):
        raw_mode = raw_mode.strip()
        if not raw_mode:
            continue
        modes.append(normalize_retrieval_mode(raw_mode).value)

    if not modes:
        raise ValueError("At least one retrieval mode is required.")

    return modes


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class BenchmarkRunner:
    def __init__(self, output_root: Path = Path("runs")):
        self.output_root = output_root

    def run(
        self,
        dataset_path: Path,
        benchmark: str,
        modes: List[str],
        limit: int | None = None,
        evaluation_backend: str | None = None,
    ) -> Dict[str, Any]:
        benchmark_dir = self.output_root / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        evaluation_backend = normalize_evaluation_backend(evaluation_backend)

        mode_summaries = []
        for mode in modes:
            mode = normalize_retrieval_mode(mode).value
            experiment_name = f"{benchmark}_{mode}"
            logger.info("Running benchmark '%s' mode '%s'", benchmark, mode)

            runner = ExperimentRunner(
                output_root=self.output_root,
                retrieval_mode=mode,
                evaluation_backend=evaluation_backend,
            )
            summary = runner.run(dataset_path, experiment_name, limit=limit)
            summary["retrieval_mode"] = mode
            summary["experiment"] = experiment_name
            mode_summaries.append(summary)

        comparison = self._comparison_summary(
            benchmark=benchmark,
            dataset_path=dataset_path,
            modes=modes,
            mode_summaries=mode_summaries,
            limit=limit,
            evaluation_backend=evaluation_backend,
        )
        write_json(benchmark_dir / "comparison.json", comparison)
        write_json(benchmark_dir / "summary.json", comparison)
        return comparison

    def _comparison_summary(
        self,
        benchmark: str,
        dataset_path: Path,
        modes: List[str],
        mode_summaries: List[Dict[str, Any]],
        limit: int | None,
        evaluation_backend: str,
    ) -> Dict[str, Any]:
        ranked = sorted(
            mode_summaries,
            key=lambda item: item.get("overall_rag_score", 0.0),
            reverse=True,
        )

        best = ranked[0] if ranked else None
        comparison = {
            "benchmark": benchmark,
            "dataset_path": str(dataset_path),
            "limit": limit,
            "modes": modes,
            "evaluation_backend": evaluation_backend,
            "experiments": [summary["experiment"] for summary in mode_summaries],
            "best_by_overall_rag_score": best,
            "results": mode_summaries,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        if best:
            for key in [
                "questions",
                "overall_rag_score",
                "avg_context_relevance",
                "avg_groundedness",
                "avg_answer_relevance",
                "avg_latency_seconds",
            ]:
                comparison[key] = best.get(key)
            comparison["best_retrieval_mode"] = best.get("retrieval_mode")

        return comparison
