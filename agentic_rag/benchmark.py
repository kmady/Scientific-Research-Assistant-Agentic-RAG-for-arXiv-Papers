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

HIGHER_IS_BETTER_METRICS = [
    "overall_rag_score",
    "avg_context_relevance",
    "avg_groundedness",
    "avg_answer_relevance",
]
LOWER_IS_BETTER_METRICS = [
    "avg_latency_seconds",
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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
        answer_mode: str | None = None,
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
                answer_mode=answer_mode,
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
            answer_mode=answer_mode,
        )
        write_json(benchmark_dir / "comparison.json", comparison)
        write_json(benchmark_dir / "summary.json", comparison)
        write_text(benchmark_dir / "report.md", self._report_markdown(comparison))
        return comparison

    def _comparison_summary(
        self,
        benchmark: str,
        dataset_path: Path,
        modes: List[str],
        mode_summaries: List[Dict[str, Any]],
        limit: int | None,
        evaluation_backend: str,
        answer_mode: str | None,
    ) -> Dict[str, Any]:
        ranked = sorted(
            mode_summaries,
            key=lambda item: item.get("overall_rag_score", 0.0),
            reverse=True,
        )

        best = ranked[0] if ranked else None
        lowest_latency = min(
            mode_summaries,
            key=lambda item: item.get("avg_latency_seconds", float("inf")),
            default=None,
        )
        comparison = {
            "benchmark": benchmark,
            "dataset_path": str(dataset_path),
            "limit": limit,
            "modes": modes,
            "evaluation_backend": evaluation_backend,
            "answer_mode": answer_mode or "agentic",
            "experiments": [summary["experiment"] for summary in mode_summaries],
            "best_by_overall_rag_score": best,
            "best_by_lowest_latency": lowest_latency,
            "metric_winners": self._metric_winners(mode_summaries),
            "question_type_winners": self._question_type_winners(mode_summaries),
            "ranked_results": ranked,
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

    def _metric_winners(self, mode_summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        winners = {}
        for metric in HIGHER_IS_BETTER_METRICS:
            winner = max(
                mode_summaries,
                key=lambda item: item.get(metric, float("-inf")),
                default=None,
            )
            if winner:
                winners[metric] = {
                    "retrieval_mode": winner.get("retrieval_mode"),
                    "experiment": winner.get("experiment"),
                    "value": winner.get(metric),
                    "direction": "higher_is_better",
                }

        for metric in LOWER_IS_BETTER_METRICS:
            winner = min(
                mode_summaries,
                key=lambda item: item.get(metric, float("inf")),
                default=None,
            )
            if winner:
                winners[metric] = {
                    "retrieval_mode": winner.get("retrieval_mode"),
                    "experiment": winner.get("experiment"),
                    "value": winner.get(metric),
                    "direction": "lower_is_better",
                }

        return winners

    def _question_type_winners(self, mode_summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        question_types = sorted({
            question_type
            for summary in mode_summaries
            for question_type in summary.get("by_question_type", {}).keys()
        })
        winners = {}
        for question_type in question_types:
            candidates = [
                summary
                for summary in mode_summaries
                if question_type in summary.get("by_question_type", {})
            ]
            winner = max(
                candidates,
                key=lambda item: item["by_question_type"][question_type].get("overall_rag_score", 0.0),
                default=None,
            )
            if not winner:
                continue
            metrics = winner["by_question_type"][question_type]
            winners[question_type] = {
                "retrieval_mode": winner.get("retrieval_mode"),
                "experiment": winner.get("experiment"),
                "overall_rag_score": metrics.get("overall_rag_score"),
                "questions": metrics.get("questions"),
            }

        return winners

    def _report_markdown(self, comparison: Dict[str, Any]) -> str:
        lines = [
            f"# Benchmark Report: {comparison['benchmark']}",
            "",
            f"- Dataset: `{comparison['dataset_path']}`",
            f"- Evaluation backend: `{comparison['evaluation_backend']}`",
            f"- Limit: `{comparison.get('limit')}`",
            f"- Completed at: `{comparison['completed_at']}`",
            "",
            "## Winners",
            "",
        ]

        for metric, winner in comparison.get("metric_winners", {}).items():
            value = winner.get("value", 0.0)
            lines.append(
                f"- `{metric}`: `{winner.get('retrieval_mode')}` "
                f"({float(value):.4f}, {winner.get('direction')})"
            )

        question_type_winners = comparison.get("question_type_winners", {})
        if question_type_winners:
            lines.extend([
                "",
                "## Winners by Question Type",
                "",
                "| Question type | Best mode | Overall | Questions |",
                "|---|---|---:|---:|",
            ])
            for question_type, winner in question_type_winners.items():
                lines.append(
                    "| "
                    f"{question_type} | "
                    f"{winner.get('retrieval_mode', '')} | "
                    f"{winner.get('overall_rag_score', 0.0):.4f} | "
                    f"{winner.get('questions', 0)} |"
                )

        lines.extend([
            "",
            "## Results",
            "",
            "| Rank | Mode | Overall | Context | Groundedness | Answer | Latency (s) |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ])

        for rank, row in enumerate(comparison.get("ranked_results", []), start=1):
            lines.append(
                "| "
                f"{rank} | "
                f"{row.get('retrieval_mode', '')} | "
                f"{row.get('overall_rag_score', 0.0):.4f} | "
                f"{row.get('avg_context_relevance', 0.0):.4f} | "
                f"{row.get('avg_groundedness', 0.0):.4f} | "
                f"{row.get('avg_answer_relevance', 0.0):.4f} | "
                f"{row.get('avg_latency_seconds', 0.0):.4f} |"
            )

        lines.append("")
        return "\n".join(lines)
