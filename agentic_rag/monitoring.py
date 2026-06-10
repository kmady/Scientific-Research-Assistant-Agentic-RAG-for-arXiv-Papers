from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
from typing import Optional

# LLM metrics
llm_latency_seconds = Histogram('llm_latency_seconds', 'Latency of LLM chat calls in seconds', ['model'])
llm_requests_total = Counter('llm_requests_total', 'Total LLM requests', ['model', 'status'])

# Agent metrics
agent_step_latency_seconds = Histogram('agent_step_latency_seconds', 'Latency per agent loop step in seconds')
agent_loops_total = Counter('agent_loops_total', 'Total number of agent loop iterations', [])

# System metrics (placeholder)
try:
    process_cpu_seconds = Gauge('process_cpu_seconds', 'Process CPU seconds')
except ValueError:
    # Metric may already be registered in the global registry (e.g., when reloading modules)
    process_cpu_seconds = None


def start_metrics_server(port: int = 8000) -> None:
    """Start Prometheus metrics HTTP server on given port (non-blocking)."""
    # start_http_server spawns a background thread; calling it is sufficient.
    start_http_server(port)


def observe_llm_call(model: Optional[str], duration: float, success: bool = True) -> None:
    if not model:
        model = 'unknown'
    llm_latency_seconds.labels(model=model).observe(duration)
    status = 'success' if success else 'error'
    llm_requests_total.labels(model=model, status=status).inc()


def observe_agent_step(duration: float) -> None:
    agent_step_latency_seconds.observe(duration)
    agent_loops_total.inc()
