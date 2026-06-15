from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)
_metrics_server_started = False

# LLM metrics
llm_latency_seconds = Histogram('llm_latency_seconds', 'Latency of LLM chat calls in seconds', ['model'])
llm_requests_total = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])

# Retrieval metrics
retrieval_latency_seconds = Histogram(
    'retrieval_latency_seconds',
    'Latency of retrieval calls in seconds',
    ['mode'],
)
retrieval_requests_total = Counter(
    'retrieval_requests_total',
    'Total retrieval requests',
    ['mode', 'status'],
)
retrieved_chunks_count = Histogram(
    'retrieved_chunks_count',
    'Number of chunks returned by retrieval calls',
    ['mode'],
    buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34),
)
reranker_latency_seconds = Histogram(
    'reranker_latency_seconds',
    'Latency of reranker calls in seconds',
    ['model', 'status'],
)

# Agent metrics
agent_step_latency_seconds = Histogram('agent_step_latency_seconds', 'Latency per agent loop step in seconds')
agent_loops_total = Counter('agent_loops_total', 'Total number of agent loop iterations', [])

# System metrics (placeholder)
try:
    process_cpu_seconds = Gauge('process_cpu_seconds', 'Process CPU seconds')
except ValueError:
    # Metric may already be registered in the global registry (e.g., when reloading modules)
    process_cpu_seconds = None


def start_metrics_server(port: int = 8000) -> bool:
    """Start Prometheus metrics HTTP server on given port (non-blocking)."""
    global _metrics_server_started
    if _metrics_server_started:
        return True
    # start_http_server spawns a background thread; calling it is sufficient.
    try:
        start_http_server(port)
        _metrics_server_started = True
        return True
    except OSError as exc:
        if getattr(exc, "errno", None) == 98:
            logger.warning("Prometheus metrics server already running or port %s is in use.", port)
            _metrics_server_started = True
            return True
        if isinstance(exc, PermissionError):
            logger.warning("Prometheus metrics server could not bind to port %s: %s", port, exc)
            return False
        raise


def observe_llm_call(model: Optional[str], duration: float, success: bool = True) -> None:
    if not model:
        model = 'unknown'
    llm_latency_seconds.labels(model=model).observe(duration)
    status = 'success' if success else 'error'
    llm_requests_total.labels(model=model, status=status).inc()


def observe_retrieval(mode: Optional[str], duration: float, success: bool = True, chunk_count: int = 0) -> None:
    if not mode:
        mode = 'unknown'
    status = 'success' if success else 'error'
    retrieval_latency_seconds.labels(mode=mode).observe(duration)
    retrieval_requests_total.labels(mode=mode, status=status).inc()
    if success:
        retrieved_chunks_count.labels(mode=mode).observe(chunk_count)


def observe_reranker(model: Optional[str], duration: float, success: bool = True) -> None:
    if not model:
        model = 'unknown'
    status = 'success' if success else 'error'
    reranker_latency_seconds.labels(model=model, status=status).observe(duration)


def observe_agent_step(duration: float) -> None:
    agent_step_latency_seconds.observe(duration)
    agent_loops_total.inc()
