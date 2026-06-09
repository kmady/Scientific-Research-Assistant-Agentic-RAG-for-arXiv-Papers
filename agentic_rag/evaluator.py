import json
import logging
from typing import List, Dict, Any
from agentic_rag.llm import get_llm_client

logger = logging.getLogger(__name__)

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

class RAGEvaluator:
    def __init__(self):
        self.llm = get_llm_client()

    def evaluate(self, query: str, retrieved_chunks: List[Dict[str, Any]], answer: str) -> Dict[str, Any]:
        """Runs the RAG Triad evaluation on a query, its retrieved context, and the generated answer."""
        logger.info("Starting RAG Triad evaluation...")
        
        # Format context chunks
        context_str = ""
        for i, c in enumerate(retrieved_chunks):
            context_str += f"--- Chunk {i+1} (Paper: {c.get('title', 'Unknown')}, Sec: {c.get('section', 'Unknown')}) ---\n"
            context_str += c.get("text", "") + "\n\n"
            
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
