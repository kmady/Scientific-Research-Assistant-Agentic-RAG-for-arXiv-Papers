from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard.loaders import (
    METRIC_COLUMNS,
    discover_experiments,
    load_experiments,
    results_dataframe,
    summaries_dataframe,
)


RUNS_DIR = Path("runs")

st.set_page_config(
    page_title="RAG Quality Dashboard",
    page_icon="R",
    layout="wide",
)


def score_delta(current: float, baseline: float) -> str:
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def format_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def render_empty_state() -> None:
    st.title("RAG Quality Dashboard")
    st.info("No experiment results found yet.")
    st.code(
        "python -m agentic_rag.cli eval-run "
        "--dataset data/eval/questions.jsonl --experiment baseline",
        language="bash",
    )


def render_metric_cards(summary_df: pd.DataFrame) -> None:
    latest = summary_df.iloc[-1]
    baseline = summary_df.iloc[0]
    cols = st.columns(4)
    cards = [
        ("Overall", "overall_rag_score"),
        ("Groundedness", "avg_groundedness"),
        ("Context", "avg_context_relevance"),
        ("Answer", "avg_answer_relevance"),
    ]
    for col, (label, metric) in zip(cols, cards):
        current = float(latest.get(metric, 0.0))
        base = float(baseline.get(metric, 0.0))
        col.metric(label, format_score(current), score_delta(current, base))


def render_summary_charts(summary_df: pd.DataFrame) -> None:
    st.subheader("Experiment Comparison")
    available = [c for c in METRIC_COLUMNS if c in summary_df.columns]
    chart_df = summary_df[["experiment", *available]].set_index("experiment")
    st.bar_chart(chart_df)

    display_columns = ["experiment", "questions", *available, "completed_at"]
    display_columns = [c for c in display_columns if c in summary_df.columns]
    st.dataframe(summary_df[display_columns], use_container_width=True, hide_index=True)


def render_question_table(results_df: pd.DataFrame) -> None:
    st.subheader("Question-Level Scores")
    columns = [
        "experiment",
        "id",
        "question_type",
        "overall_rag_score",
        "context_relevance",
        "groundedness",
        "answer_relevance",
        "latency_seconds",
        "agent_success",
    ]
    columns = [c for c in columns if c in results_df.columns]
    st.dataframe(
        results_df[columns].sort_values(["experiment", "overall_rag_score"], ascending=[True, True]),
        use_container_width=True,
        hide_index=True,
    )


def render_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        st.caption("No retrieved chunks recorded.")
        return

    chunk_rows = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_rows.append({
            "rank": index,
            "arxiv_id": chunk.get("arxiv_id", ""),
            "section": chunk.get("section", ""),
            "pages": f"{chunk.get('page_start', '')}-{chunk.get('page_end', '')}",
            "hybrid_score": chunk.get("hybrid_score"),
            "rerank_score": chunk.get("rerank_score"),
        })
    st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)

    for index, chunk in enumerate(chunks, start=1):
        label = f"Chunk {index}: {chunk.get('section', 'Unknown section')}"
        with st.expander(label):
            st.caption(f"{chunk.get('title', '')} | Page {chunk.get('page_start', '')}-{chunk.get('page_end', '')}")
            st.write(chunk.get("text", ""))


def render_answer_inspector(results_df: pd.DataFrame) -> None:
    st.subheader("Answer Inspection")
    if results_df.empty:
        st.caption("No question results available.")
        return

    experiments = sorted(results_df["experiment"].unique())
    selected_experiment = st.selectbox("Experiment", experiments)
    scoped = results_df[results_df["experiment"] == selected_experiment]

    question_labels = [
        f"{row.id} | {row.question[:110]}"
        for row in scoped.itertuples()
    ]
    selected_label = st.selectbox("Question", question_labels)
    selected_id = selected_label.split(" | ", 1)[0]
    record = scoped[scoped["id"] == selected_id].iloc[0]

    cols = st.columns(4)
    cols[0].metric("Overall", format_score(record.get("overall_rag_score")))
    cols[1].metric("Context", format_score(record.get("context_relevance")))
    cols[2].metric("Groundedness", format_score(record.get("groundedness")))
    cols[3].metric("Answer", format_score(record.get("answer_relevance")))

    st.markdown("**Question**")
    st.write(record.get("question", ""))

    st.markdown("**Generated Answer**")
    st.write(record.get("answer", ""))

    st.markdown("**Expected Topics**")
    st.write(", ".join(record.get("expected_topics", [])) or "None recorded.")

    st.markdown("**Retrieved Context**")
    render_chunks(record.get("retrieved_chunks", []))

    with st.expander("Agent Steps"):
        st.json(record.get("agent_steps", []))


def render_config_view(experiments: List[Dict[str, Any]]) -> None:
    st.subheader("Configuration")
    names = [experiment["name"] for experiment in experiments]
    selected = st.selectbox("Experiment config", names)
    experiment = next(item for item in experiments if item["name"] == selected)
    st.json(experiment.get("config", {}))


def main() -> None:
    experiments_available = discover_experiments(RUNS_DIR)
    if not experiments_available:
        render_empty_state()
        return

    st.title("RAG Quality Dashboard")

    selected = st.sidebar.multiselect(
        "Experiments",
        experiments_available,
        default=experiments_available,
    )
    if not selected:
        st.warning("Select at least one experiment.")
        return

    experiments = load_experiments(RUNS_DIR, selected)
    summary_df = summaries_dataframe(experiments)
    result_df = results_dataframe(experiments)

    render_metric_cards(summary_df)

    tabs = st.tabs([
        "Overview",
        "Questions",
        "Answer Inspection",
        "Configuration",
    ])

    with tabs[0]:
        render_summary_charts(summary_df)

    with tabs[1]:
        if result_df.empty:
            st.caption("No per-question results available.")
        else:
            render_question_table(result_df)

    with tabs[2]:
        render_answer_inspector(result_df)

    with tabs[3]:
        render_config_view(experiments)


if __name__ == "__main__":
    main()
