import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


METRIC_COLUMNS = [
    "overall_rag_score",
    "avg_context_relevance",
    "avg_groundedness",
    "avg_answer_relevance",
    "avg_latency_seconds",
]


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def discover_experiments(runs_dir: Path) -> List[str]:
    if not runs_dir.exists():
        return []
    return sorted(
        path.name
        for path in runs_dir.iterdir()
        if path.is_dir() and (path / "summary.json").exists()
    )


def load_experiment(runs_dir: Path, experiment: str) -> Dict[str, Any]:
    experiment_dir = runs_dir / experiment
    return {
        "name": experiment,
        "path": experiment_dir,
        "summary": read_json(experiment_dir / "summary.json"),
        "comparison": read_json(experiment_dir / "comparison.json"),
        "config": read_json(experiment_dir / "config.json"),
        "results": read_jsonl(experiment_dir / "results.jsonl"),
    }


def load_experiments(runs_dir: Path, names: List[str]) -> List[Dict[str, Any]]:
    return [load_experiment(runs_dir, name) for name in names]


def summaries_dataframe(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for experiment in experiments:
        row = {"experiment": experiment["name"]}
        row.update(experiment.get("summary", {}))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["experiment", *METRIC_COLUMNS])
    return pd.DataFrame(rows)


def results_dataframe(experiments: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for experiment in experiments:
        for record in experiment.get("results", []):
            evaluation = record.get("evaluation", {})
            rows.append({
                "experiment": experiment["name"],
                "id": record.get("id", ""),
                "question": record.get("question", ""),
                "question_type": record.get("question_type", ""),
                "overall_rag_score": evaluation.get("overall_rag_score", 0.0),
                "context_relevance": evaluation.get("context_relevance", {}).get("score", 0.0),
                "groundedness": evaluation.get("groundedness", {}).get("score", 0.0),
                "answer_relevance": evaluation.get("answer_relevance", {}).get("score", 0.0),
                "latency_seconds": record.get("latency_seconds", 0.0),
                "agent_success": record.get("agent_success", False),
                "answer": record.get("answer", ""),
                "retrieved_chunks": record.get("retrieved_chunks", []),
                "agent_steps": record.get("agent_steps", []),
                "expected_topics": record.get("expected_topics", []),
            })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
