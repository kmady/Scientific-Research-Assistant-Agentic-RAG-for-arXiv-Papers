import json
import logging
import time
from typing import List, Dict, Any, Tuple
from agentic_rag import config
from agentic_rag.llm import get_llm_client, LLMClient
from agentic_rag.search import ArxivSearchAgent
from agentic_rag.pdf_parser import PDFProcessor
from agentic_rag.vector_db import VectorStore
from agentic_rag import monitoring

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Advanced Scientific Research Assistant with access to the arXiv repository.
Your task is to answer user queries with precise scientific depth, comparisons of methods, extraction of contributions, limitations, and results, backed by exact citations.

You have access to the following tools:

1. `search_arxiv`: Search for papers on arXiv.
   - Action Input: `{"query": "search query", "limit": 5}`
   - Observation: List of matching paper titles, abstracts, and arXiv IDs.

2. `download_and_index`: Download a PDF from arXiv and index it into your local vector database.
   - Action Input: `{"arxiv_id": "clean_arxiv_id"}`
   - Observation: Success/Failure confirmation and number of chunks added.

3. `retrieve_context`: Query the local vector store for highly relevant chunks.
   - Action Input: `{"query": "retrieval query", "top_k": 10}`
   - Observation: Content-rich chunks containing text, paper titles, sections, and page numbers.

4. `answer_user`: Formulate the final answer to the user.
   - Action Input: `{"synthesis": "your comprehensive scientific answer"}`
   - Observation: None. This ends the loop.

INSTRUCTIONS:
- You must always respond in JSON format with three fields:
  {
    "thought": "Your reasoning about what tool to use and why",
    "action": "search_arxiv" | "download_and_index" | "retrieve_context" | "answer_user",
    "action_input": { ... } // arguments matching the tool description
  }
- Cite papers explicitly using the format: `[ArXiv:XXXX.YYYY, Sec: Section Name, p. Z]`.
- Always check if a paper is already indexed. You can check the vector store first.
- If you need to answer a question that requires reading a paper in detail, you MUST call `download_and_index` first, then call `retrieve_context` to fetch the details.
- Be thorough. Compare methods, analyze limitations, and extract core findings.
"""

class AgenticOrchestrator:
    def __init__(self):
        # Start metrics server (exposes /metrics on port 8000)
        try:
            monitoring.start_metrics_server(port=int(config.__dict__.get('METRICS_PORT', 8000)))
            logger.info("Prometheus metrics server started on port %s", config.__dict__.get('METRICS_PORT', 8000))
        except Exception:
            logger.exception("Failed to start Prometheus metrics server")

        self.llm = get_llm_client()
        self.search_agent = ArxivSearchAgent()
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.max_steps = 10

    def run(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"Starting agent loop for query: {user_query}")
        
        # Get list of already indexed papers
        indexed_ids = self.vector_store.get_indexed_arxiv_ids()
        
        history: List[Dict[str, str]] = []
        
        # Add initial user context
        current_state = f"User Query: {user_query}\n\nCurrently indexed paper IDs in local Vector Store: {indexed_ids}"
        
        steps_taken = []
        
        for step in range(self.max_steps):
            logger.info(f"Agent Loop - Step {step + 1}/{self.max_steps}")
            
            # Format history for LLM
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": current_state}
            ]
            
            try:
                start_t = time.time()
                response = self.llm.chat(messages, temperature=0.1, response_json=True)
                duration = time.time() - start_t
                model_name = getattr(self.llm, 'model', config.LLM_PROVIDER)
                monitoring.observe_llm_call(model=model_name, duration=duration, success=True)

                response_text = response.content
                logger.info(f"Agent Response: {response_text}")
                
                action_data = json.loads(response_text)
            except Exception as e:
                # Record failed LLM attempt
                try:
                    model_name = getattr(self.llm, 'model', config.LLM_PROVIDER)
                    monitoring.observe_llm_call(model=model_name, duration=(time.time() - start_t) if 'start_t' in locals() else 0.0, success=False)
                except Exception:
                    pass

                error_msg = f"Failed to parse your response as JSON. Make sure you return valid JSON. Error: {str(e)}"
                logger.warning(error_msg)
                current_state += f"\n\nObservation (System): {error_msg}"
                steps_taken.append({"thought": "Failed to parse JSON", "error": str(e)})
                continue
                
            thought = action_data.get("thought", "")
            action = action_data.get("action", "")
            action_input = action_data.get("action_input", {})
            
            steps_taken.append({
                "step": step + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input
            })
            
            if action == "answer_user":
                logger.info("Agent decided to answer user. Terminating loop.")
                synthesis = action_input.get("synthesis", "")
                if not synthesis:
                    synthesis = action_input.get("answer", "") or action_input.get("response", "")
                if not synthesis:
                    logger.warning("answer_user action did not include a synthesis. Falling back to thought text.")
                    synthesis = thought
                return {
                    "answer": synthesis,
                    "steps": steps_taken,
                    "success": True
                }
                
            # Execute tool
            observation = ""
            try:
                if action == "search_arxiv":
                    query = action_input.get("query", "")
                    limit = int(action_input.get("limit", 5))
                    results = self.search_agent.search(query, max_results=limit)
                    
                    # Clean/format results for the LLM
                    formatted_results = []
                    for r in results:
                        formatted_results.append({
                            "id": r["id"],
                            "title": r["title"],
                            "authors": r["authors"],
                            "published": r["published"],
                            "summary": r["summary"][:300] + "..."  # abstract snippet
                        })
                    observation = json.dumps(formatted_results, indent=2)
                    
                elif action == "download_and_index":
                    arxiv_id = action_input.get("arxiv_id", "")
                    # Fetch metadata to get details if we don't have them
                    metadata_list = self.search_agent.fetch_by_ids([arxiv_id])
                    if not metadata_list:
                        observation = f"Failed: Could not find paper with arXiv ID {arxiv_id} on arXiv API."
                    else:
                        metadata = metadata_list[0]
                        pdf_path = self.search_agent.download_pdf(arxiv_id, pdf_link=metadata.get("pdf_link"))
                        if pdf_path:
                            chunks = self.pdf_processor.process_paper(pdf_path, metadata)
                            self.vector_store.add_chunks(chunks)
                            observation = f"Success: Downloaded and indexed paper '{metadata['title']}' ({arxiv_id}). Added {len(chunks)} chunks to vector store."
                        else:
                            observation = f"Failed: Could not download PDF for {arxiv_id}. The arXiv link might be blocked or require a CAPTCHA."
                            
                elif action == "retrieve_context":
                    query = action_input.get("query", "")
                    top_k = int(action_input.get("top_k", 8))
                    retrieved_chunks = self.vector_store.hybrid_search(query, top_k=top_k)
                    
                    formatted_chunks = []
                    for c in retrieved_chunks:
                        formatted_chunks.append({
                            "arxiv_id": c["arxiv_id"],
                            "title": c["title"],
                            "section": c["section"],
                            "page": f"{c['page_start']}-{c['page_end']}",
                            "text": c["text"]
                        })
                    observation = json.dumps(formatted_chunks, indent=2)
                else:
                    observation = f"Error: Unknown action '{action}'."
                    
            except Exception as e:
                observation = f"Error executing tool '{action}': {str(e)}"
                logger.error(observation, exc_info=True)
                
            logger.info(f"Observation length: {len(observation)} chars")
            
            # Update state with step execution details
            current_state += f"\n\nStep {step + 1} Action: {action}\nAction Input: {json.dumps(action_input)}\nObservation: {observation}"
            
        # If we exited without answering, force synthesis
        logger.warning("Max steps reached without explicit answer. Forcing synthesis.")
        force_prompt = [
            {"role": "system", "content": "Synthesize a final answer based on all the steps and observations collected. Answer the user query using the information available, and cite references where appropriate."},
            {"role": "user", "content": current_state}
        ]
        try:
            start_t = time.time()
            final_resp = self.llm.chat(force_prompt, temperature=0.2)
            duration = time.time() - start_t
            model_name = getattr(self.llm, 'model', config.LLM_PROVIDER)
            monitoring.observe_llm_call(model=model_name, duration=duration, success=True)
            return {
                "answer": final_resp.content,
                "steps": steps_taken,
                "success": False,
                "error": "Max steps reached"
            }
        except Exception as e:
            try:
                model_name = getattr(self.llm, 'model', config.LLM_PROVIDER)
                monitoring.observe_llm_call(model=model_name, duration=(time.time() - start_t) if 'start_t' in locals() else 0.0, success=False)
            except Exception:
                pass
            return {
                "answer": "An error occurred during final synthesis.",
                "steps": steps_taken,
                "success": False,
                "error": str(e)
            }
        
    def batch_ingest_by_query(self, query: str, limit: int = 10):
        """Helper to bulk-ingest papers by search query (for populating the database)."""
        logger.info(f"Bulk ingesting up to {limit} papers matching query: {query}")
        papers = self.search_agent.search(query, max_results=limit)
        
        indexed_ids = self.vector_store.get_indexed_arxiv_ids()
        
        success_count = 0
        for paper in papers:
            arxiv_id = paper["id"]
            if arxiv_id in indexed_ids:
                logger.info(f"Paper {arxiv_id} already indexed. Skipping.")
                continue
                
            pdf_path = self.search_agent.download_pdf(arxiv_id, pdf_link=paper.get("pdf_link"))
            if pdf_path:
                try:
                    chunks = self.pdf_processor.process_paper(pdf_path, paper)
                    self.vector_store.add_chunks(chunks)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Error processing paper {arxiv_id}: {e}")
            
        logger.info(f"Bulk ingest complete. Successfully indexed {success_count}/{len(papers)} papers.")
        return success_count
