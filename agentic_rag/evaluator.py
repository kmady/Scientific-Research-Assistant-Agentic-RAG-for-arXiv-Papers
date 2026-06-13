import json
import logging
from typing import List, Dict, Any
from agentic_rag import config
from agentic_rag.llm import get_llm_client

logger = logging.getLogger(__name__)

SUPPORTED_EVALUATION_BACKENDS = {"llm_judge", "ragas", "deepeval"}

EVAL_PROMPTS = {
    "context_relevance": """You are an independent scientific judge. Your task is to evaluate the relevance of the retrieved context chunks to the user's query.

User Query: {query}

Retrieved Context Chunks:
{context}

Review the retrieved chunks. For each chunk, determine if it contains information directly helpful for answering the user's query.
Respond in JSON format with two fields:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means all chunks are highly relevant, and 0.0 means none of them are relevant>,
  "reason": "Provide a brief, 2-3 sentence explanation of your scoring."
}}
""",

    "groundedness": """You are an independent scientific judge. Your task is to evaluate the groundedness (faithfulness) of the generated answer. The answer must be fully supported by the retrieved context. No external knowledge or hallucinations are allowed.

Retrieved Context Chunks:
{context}

Generated Answer:
{answer}

Check if every claim made in the generated answer is directly backed by the retrieved context chunks. If there are claims in the answer that cannot be found in the context, the score must be low.
Respond in JSON format with two fields:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means the answer is 100% supported by the context, and 0.0 means the answer is completely unsupported or contains hallucinations>,
  "reason": "Explain your score, pointing out any specific claims in the answer that are not supported by the context."
}}
""",

    "answer_relevance": """You are an independent scientific judge. Your task is to evaluate how well the generated answer addresses the user's query. It should be helpful, directly answer the question, and not be vague or copy-pasted nonsense.

User Query: {query}

Generated Answer:
{answer}

Rate how relevant and complete the answer is relative to the user query.
Respond in JSON format with two fields:
{{
  "score": <float between 0.0 and 1.0, where 1.0 means the answer directly, clearly, and fully addresses the user query, and 0.0 means the answer is completely irrelevant or fails to answer the question>,
  "reason": "Provide a brief explanation of your scoring."
}}
"""
}

def normalize_evaluation_backend(backend: str | None) -> str:
    backend = (backend or config.EVALUATION_BACKEND).strip().lower()
    if backend not in SUPPORTED_EVALUATION_BACKENDS:
        valid = ", ".join(sorted(SUPPORTED_EVALUATION_BACKENDS))
        raise ValueError(f"Invalid evaluation backend '{backend}'. Expected one of: {valid}")
    return backend


class RAGEvaluator:
    def __init__(self, backend: str | None = None):
        self.backend = normalize_evaluation_backend(backend)
        self.llm = get_llm_client()

    def evaluate(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Runs the RAG Triad evaluation on a query, its retrieved context, and the generated answer."""
        if self.backend == "ragas":
            return self._evaluate_with_ragas(query, retrieved_chunks, answer)
        if self.backend == "deepeval":
            return self._evaluate_with_deepeval(query, retrieved_chunks, answer)

        return self._evaluate_with_llm_judge(query, retrieved_chunks, answer)

    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        context_str = ""
        for i, c in enumerate(retrieved_chunks):
            context_str += f"--- Chunk {i+1} (Paper: {c.get('title', 'Unknown')}, Sec: {c.get('section', 'Unknown')}) ---\n"
            context_str += c.get("text", "") + "\n\n"
        return context_str

    def _backend_error_result(self, backend: str, message: str) -> Dict[str, Any]:
        result = {
            "context_relevance": {"score": 0.0, "reason": message},
            "groundedness": {"score": 0.0, "reason": message},
            "answer_relevance": {"score": 0.0, "reason": message},
            "overall_rag_score": 0.0,
            "evaluation_backend": backend,
            "backend_error": message,
        }
        return result

    def _evaluate_with_llm_judge(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Runs the local LLM-judge RAG Triad evaluation."""
        logger.info("Starting RAG Triad evaluation...")
        context_str = self._format_context(retrieved_chunks)
        results = {}
        
        # 1. Context Relevance
        try:
            prompt = EVAL_PROMPTS["context_relevance"].format(query=query, context=context_str)
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, response_json=True)
            results["context_relevance"] = json.loads(resp.content)
        except Exception as e:
            logger.error(f"Failed to evaluate context relevance: {e}")
            results["context_relevance"] = {"score": 0.0, "reason": f"Evaluation error: {str(e)}"}
            
        # 2. Groundedness (Faithfulness)
        try:
            prompt = EVAL_PROMPTS["groundedness"].format(context=context_str, answer=answer)
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, response_json=True)
            results["groundedness"] = json.loads(resp.content)
        except Exception as e:
            logger.error(f"Failed to evaluate groundedness: {e}")
            results["groundedness"] = {"score": 0.0, "reason": f"Evaluation error: {str(e)}"}

        # 3. Answer Relevance
        try:
            prompt = EVAL_PROMPTS["answer_relevance"].format(query=query, answer=answer)
            resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, response_json=True)
            results["answer_relevance"] = json.loads(resp.content)
        except Exception as e:
            logger.error(f"Failed to evaluate answer relevance: {e}")
            results["answer_relevance"] = {"score": 0.0, "reason": f"Evaluation error: {str(e)}"}
            
        # Compute overall RAG score (average of the three)
        scores = [
            results["context_relevance"]["score"],
            results["groundedness"]["score"],
            results["answer_relevance"]["score"]
        ]
        results["overall_rag_score"] = sum(scores) / len(scores) if scores else 0.0
        results["evaluation_backend"] = "llm_judge"
        
        return results

    def _evaluate_with_ragas(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Optional RAGAS integration placeholder.

        RAGAS is intentionally optional so the default project remains lightweight.
        Install and wire the provider dependencies before using this backend for
        publication-quality experiments.
        """
        try:
            import ragas  # noqa: F401
        except ImportError:
            return self._backend_error_result(
                "ragas",
                "RAGAS backend requested, but ragas is not installed. Install optional evaluation dependencies before using EVALUATION_BACKEND=ragas.",
            )

        return self._backend_error_result(
            "ragas",
            "RAGAS backend scaffold is available, but metric execution has not been configured for this project yet.",
        )

    def _evaluate_with_deepeval(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Optional DeepEval integration placeholder."""
        try:
            import deepeval  # noqa: F401
        except ImportError:
            return self._backend_error_result(
                "deepeval",
                "DeepEval backend requested, but deepeval is not installed. Install optional evaluation dependencies before using EVALUATION_BACKEND=deepeval.",
            )

        return self._backend_error_result(
            "deepeval",
            "DeepEval backend scaffold is available, but metric execution has not been configured for this project yet.",
        )
        
    def generate_report_markdown(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str, eval_results: Dict[str, Any]) -> str:
        """Helper to generate a clean, formatted Markdown report of the evaluation."""
        report = []
        report.append(f"# RAG Triad Evaluation Report\n")
        report.append(f"**Query**: {query}\n")
        report.append(f"## Scores Summary\n")
        report.append(f"| Metric | Score | Detail |")
        report.append(f"|---|---|---|")
        report.append(f"| Context Relevance | `{eval_results['context_relevance']['score']:.2f}` | {eval_results['context_relevance']['reason']} |")
        report.append(f"| Groundedness (Faithfulness) | `{eval_results['groundedness']['score']:.2f}` | {eval_results['groundedness']['reason']} |")
        report.append(f"| Answer Relevance | `{eval_results['answer_relevance']['score']:.2f}` | {eval_results['answer_relevance']['reason']} |")
        report.append(f"| **Overall RAG Triad Score** | `**{eval_results['overall_rag_score']:.2f}**` | **Average of all scores** |\n")
        
        report.append(f"## Retrieved Sources ({len(retrieved_chunks)} chunks)")
        for i, c in enumerate(retrieved_chunks):
            report.append(f"- **[{c.get('arxiv_id')}]** {c.get('title')} - *Section: {c.get('section')}* (Page {c.get('page_start')}-{c.get('page_end')})")
            
        return "\n".join(report)
