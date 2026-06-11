# Monitoring Stack

This directory contains a lightweight Docker-based monitoring stack for the project.

## Start monitoring

Run from the project root:

```bash
docker compose -f monitoring/docker-compose.yml up -d
```

This will start:
- Prometheus on `http://localhost:9090`
- Grafana on `http://localhost:3000`

## What is configured

- `monitoring/prometheus.yml`: scrapes the local app metrics at `host.docker.internal:8000/metrics`
- `monitoring/prometheus_rules.yml`: alerting rules for latency and LLM error spikes
- `monitoring/grafana/provisioning`: auto-provisions a Prometheus datasource and dashboard
- `monitoring/grafana/dashboards/agentic_rag_dashboard.json`: Grafana dashboard with key metrics

## Notes

- The application must already expose Prometheus metrics on `http://localhost:8000/metrics`.
- Grafana admin login is `admin` / `admin` by default.

## Stop monitoring

```bash
docker compose -f monitoring/docker-compose.yml down
```
