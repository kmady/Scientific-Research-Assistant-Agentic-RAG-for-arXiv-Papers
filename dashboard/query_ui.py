import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_rag.evaluator import RAGEvaluator
from agentic_rag.orchestrator import AgenticOrchestrator


st.set_page_config(
    page_title="RAG Prompt UI",
    page_icon="Q",
    layout="wide",
)


@st.cache_resource
def get_orchestrator() -> AgenticOrchestrator:
    return AgenticOrchestrator()


@st.cache_resource
def get_evaluator() -> RAGEvaluator:
    return RAGEvaluator()


def render_steps(steps: List[Dict[str, Any]]) -> None:
    if not steps:
        st.caption("No agent steps recorded.")
        return

    for step in steps:
        label = f"Step {step.get('step', '?')}: {step.get('action', 'unknown')}"
        with st.expander(label):
            st.markdown("**Thought**")
            st.write(step.get("thought", ""))
            st.markdown("**Action Input**")
            st.json(step.get("action_input", {}))


def render_chunks(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        st.caption("No retrieved chunks found.")
        return

    for index, chunk in enumerate(chunks, start=1):
        section = chunk.get("section", "Unknown section")
        arxiv_id = chunk.get("arxiv_id", "unknown")
        label = f"{index}. {arxiv_id} | {section}"
        with st.expander(label):
            st.caption(
                f"{chunk.get('title', '')} | "
                f"Pages {chunk.get('page_start', '')}-{chunk.get('page_end', '')}"
            )
            st.write(chunk.get("text", ""))


def main() -> None:
    st.title("RAG Prompt UI")

    prompt = st.text_area(
        "Prompt",
        height=140,
        placeholder="Ask a scientific question about the indexed arXiv papers...",
    )

    cols = st.columns([1, 1, 4])
    run_query = cols[0].button("Ask", type="primary", use_container_width=True)
    run_eval = cols[1].checkbox("Evaluate")

    if not run_query:
        st.info("Enter a prompt and click Ask.")
        return

    if not prompt.strip():
        st.warning("Please enter a prompt.")
        return

    orchestrator = get_orchestrator()

    with st.spinner("Running the agentic RAG pipeline..."):
        result = orchestrator.run(prompt.strip())

    st.subheader("Answer")
    st.markdown(result.get("answer", "No answer returned."))

    tabs = st.tabs(["Agent Steps", "Retrieved Context", "Evaluation"])

    with tabs[0]:
        render_steps(result.get("steps", []))

    with tabs[1]:
        with st.spinner("Retrieving supporting context..."):
            chunks = orchestrator.vector_store.hybrid_search(prompt.strip(), top_k=8)
        render_chunks(chunks)

    with tabs[2]:
        if not run_eval:
            st.caption("Enable Evaluate before asking to score this answer.")
        else:
            evaluator = get_evaluator()
            with st.spinner("Evaluating answer quality..."):
                evaluation = evaluator.evaluate(prompt.strip(), chunks, result.get("answer", ""))
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Overall", f"{evaluation.get('overall_rag_score', 0.0):.3f}")
            col_b.metric("Context", f"{evaluation.get('context_relevance', {}).get('score', 0.0):.3f}")
            col_c.metric("Groundedness", f"{evaluation.get('groundedness', {}).get('score', 0.0):.3f}")
            col_d.metric("Answer", f"{evaluation.get('answer_relevance', {}).get('score', 0.0):.3f}")
            st.json(evaluation)


if __name__ == "__main__":
    main()
