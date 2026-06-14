import json
import logging
from typing import List, Dict, Any, Callable
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

    def evaluate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str,
        expected_topics: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Runs the RAG Triad evaluation on a query, its retrieved context, and the generated answer."""
        if self.backend == "ragas":
            return self._evaluate_with_ragas(query, retrieved_chunks, answer, expected_topics)
        if self.backend == "deepeval":
            return self._evaluate_with_deepeval(query, retrieved_chunks, answer, expected_topics)

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

    def _context_texts(self, retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
        return [chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")]

    def _reference_text(self, expected_topics: List[str] | None) -> str:
        if not expected_topics:
            return ""
        return "; ".join(expected_topics)

    def _metric_result(self, score: float | None, reason: str) -> Dict[str, Any]:
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            numeric_score = 0.0
        return {"score": max(0.0, min(1.0, numeric_score)), "reason": reason}

    def _overall_score(self, results: Dict[str, Any]) -> float:
        scores = [
            results["context_relevance"]["score"],
            results["groundedness"]["score"],
            results["answer_relevance"]["score"],
        ]
        return sum(scores) / len(scores) if scores else 0.0

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
            
        results["overall_rag_score"] = self._overall_score(results)
        results["evaluation_backend"] = "llm_judge"
        
        return results

    def _evaluate_with_ragas(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str,
        expected_topics: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Run RAGAS when optional dependencies and provider config are available."""
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import answer_relevancy, context_precision, faithfulness
            from ragas.run_config import RunConfig
        except ImportError:
            return self._backend_error_result(
                "ragas",
                "RAGAS backend requested, but optional dependencies are not installed. Install requirements-eval.txt before using EVALUATION_BACKEND=ragas.",
            )

        runtime = self._ragas_runtime()
        if runtime.get("error"):
            return self._backend_error_result("ragas", runtime["error"])

        contexts = self._context_texts(retrieved_chunks)
        reference = self._reference_text(expected_topics) or query
        dataset = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [reference],
            "reference": [reference],
        })

        try:
            result = ragas_evaluate(
                dataset,
                metrics=[context_precision, faithfulness, answer_relevancy],
                llm=runtime.get("llm"),
                embeddings=runtime.get("embeddings"),
                run_config=RunConfig(
                    timeout=config.RAGAS_TIMEOUT_SECONDS,
                    max_retries=config.RAGAS_MAX_RETRIES,
                    max_workers=config.RAGAS_MAX_WORKERS,
                ),
                raise_exceptions=True,
                show_progress=False,
            )
            scores = result.to_pandas().iloc[0].to_dict()
        except Exception as exc:
            error_detail = f"{exc.__class__.__name__}: {exc!s}" if str(exc) else repr(exc)
            logger.error("RAGAS evaluation failed: %s", error_detail)
            return self._backend_error_result("ragas", f"RAGAS evaluation error: {error_detail}")

        results = {
            "context_relevance": self._metric_result(
                scores.get("context_precision"),
                "RAGAS context_precision mapped to context_relevance.",
            ),
            "groundedness": self._metric_result(
                scores.get("faithfulness"),
                "RAGAS faithfulness mapped to groundedness.",
            ),
            "answer_relevance": self._metric_result(
                scores.get("answer_relevancy"),
                "RAGAS answer_relevancy mapped to answer_relevance.",
            ),
            "evaluation_backend": "ragas",
            "raw_scores": scores,
        }
        results["overall_rag_score"] = self._overall_score(results)
        return results

    def _ragas_runtime(self) -> Dict[str, Any]:
        provider = config.LLM_PROVIDER.lower()
        if provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                from langchain_community.embeddings import OllamaEmbeddings
            except ImportError as exc:
                return {
                    "error": (
                        "RAGAS with Ollama requires compatible LangChain optional dependencies. "
                        f"Import error: {exc}"
                    )
                }

            return {
                "llm": ChatOllama(
                    model=config.OLLAMA_MODEL,
                    base_url=config.OLLAMA_HOST,
                    temperature=0.0,
                ),
                "embeddings": OllamaEmbeddings(
                    model=config.RAGAS_OLLAMA_EMBEDDING_MODEL,
                    base_url=config.OLLAMA_HOST,
                ),
            }

        if provider == "openai":
            if not config.OPENAI_API_KEY:
                return {"error": "RAGAS with OpenAI requires OPENAI_API_KEY."}
            return {"llm": None, "embeddings": None}

        return {
            "error": (
                "RAGAS requires a real evaluator model. Set LLM_PROVIDER=ollama with "
                f"OLLAMA_MODEL and RAGAS_OLLAMA_EMBEDDING_MODEL, or set LLM_PROVIDER=openai with OPENAI_API_KEY. Current provider: {provider}."
            )
        }

    def _evaluate_with_deepeval(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        answer: str,
        expected_topics: List[str] | None = None,
    ) -> Dict[str, Any]:
        """Run DeepEval when optional dependencies and provider config are available."""
        try:
            from deepeval.metrics import AnswerRelevancyMetric, ContextualRelevancyMetric, FaithfulnessMetric
            from deepeval.test_case import LLMTestCase
        except ImportError:
            return self._backend_error_result(
                "deepeval",
                "DeepEval backend requested, but optional dependencies are not installed. Install requirements-eval.txt before using EVALUATION_BACKEND=deepeval.",
            )

        test_case = LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=self._reference_text(expected_topics),
            retrieval_context=self._context_texts(retrieved_chunks),
        )
        metric_factories: Dict[str, Callable[[], Any]] = {
            "context_relevance": lambda: ContextualRelevancyMetric(threshold=0.0),
            "groundedness": lambda: FaithfulnessMetric(threshold=0.0),
            "answer_relevance": lambda: AnswerRelevancyMetric(threshold=0.0),
        }

        results: Dict[str, Any] = {"evaluation_backend": "deepeval", "raw_scores": {}}
        for target_name, factory in metric_factories.items():
            try:
                metric = factory()
                metric.measure(test_case)
                score = getattr(metric, "score", 0.0)
                reason = getattr(metric, "reason", f"DeepEval {metric.__class__.__name__} score.")
                results[target_name] = self._metric_result(score, reason)
                results["raw_scores"][metric.__class__.__name__] = {
                    "score": score,
                    "reason": reason,
                }
            except Exception as exc:
                logger.error("DeepEval metric %s failed: %s", target_name, exc)
                results[target_name] = self._metric_result(0.0, f"DeepEval metric error: {exc}")

        results["overall_rag_score"] = self._overall_score(results)
        return results
        
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
