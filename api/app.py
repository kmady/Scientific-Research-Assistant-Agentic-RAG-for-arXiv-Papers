import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agentic_rag.orchestrator import AgenticOrchestrator
from agentic_rag.vector_db import normalize_retrieval_mode


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic RAG API",
    version="0.1.0",
    description="Production HTTP wrapper around the Agentic RAG orchestrator.",
)


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    include_steps: bool = True
    retrieval_mode: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    success: bool
    steps: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class IngestRequest(BaseModel):
    ids: List[str] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    requested: int
    ingested: int
    failures: List[Dict[str, str]]


@lru_cache(maxsize=8)
def get_orchestrator(retrieval_mode: str) -> AgenticOrchestrator:
    return AgenticOrchestrator(retrieval_mode=retrieval_mode)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> Dict[str, Any]:
    orchestrator = get_orchestrator(normalize_retrieval_mode(None).value)
    return {
        "status": "ready",
        "indexed_papers": orchestrator.vector_store.get_indexed_arxiv_ids(),
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        retrieval_mode = normalize_retrieval_mode(request.retrieval_mode).value
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = get_orchestrator(retrieval_mode).run(request.prompt)
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return QueryResponse(
        answer=result.get("answer", ""),
        success=bool(result.get("success", False)),
        steps=result.get("steps", []) if request.include_steps else None,
        error=result.get("error"),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    orchestrator = get_orchestrator(normalize_retrieval_mode(None).value)
    ingested = 0
    failures: List[Dict[str, str]] = []

    for arxiv_id in request.ids:
        try:
            metadata_list = orchestrator.search_agent.fetch_by_ids([arxiv_id])
            if not metadata_list:
                failures.append({"id": arxiv_id, "reason": "arXiv ID not found"})
                continue

            metadata = metadata_list[0]
            pdf_path = orchestrator.search_agent.download_pdf(
                arxiv_id,
                pdf_link=metadata.get("pdf_link"),
            )
            if not pdf_path:
                failures.append({"id": arxiv_id, "reason": "PDF download failed"})
                continue

            chunks = orchestrator.pdf_processor.process_paper(pdf_path, metadata)
            orchestrator.vector_store.add_chunks(chunks)
            ingested += 1
        except Exception as exc:
            logger.exception("Ingestion failed for %s", arxiv_id)
            failures.append({"id": arxiv_id, "reason": str(exc)})

    return IngestResponse(
        requested=len(request.ids),
        ingested=ingested,
        failures=failures,
    )
