# Runbooks - Agentic RAG

## High LLM latency (p95)
1. Check Grafana p95 panel for affected model.
2. Inspect Prometheus `llm_requests_total` and `llm_latency_seconds` metrics.
3. Check LLM provider health (OLLAMA / OpenAI / Gemini): reachability and error rates.
4. If using local Ollama, confirm the container/service has enough CPU and memory.
5. Restart the local LLM service if needed.
6. If latency persists, switch to a smaller model in `.env` and re-run canary checks.

## LLM error spike
1. Inspect recent logs (search for errors/exceptions) using centralized logging.
2. Identify error traces and correlated requests (use trace_id if present).
3. Check API keys and rate limits for remote providers.
4. Redeploy or roll back recent code changes if errors started after a deployment.

## Vector store degraded / index issues
1. Check available disk space and index file integrity (`data/vector_store/index.faiss`).
2. Rebuild index from cached chunks if corrupted.
3. Verify retrieval correctness with sample queries.

General: always follow the escalation path and notify on-call via the configured alerting channel.
