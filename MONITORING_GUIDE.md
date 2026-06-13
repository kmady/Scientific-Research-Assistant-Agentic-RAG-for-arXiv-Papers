# Guide de Monitoring - Agentic RAG

## Problème Résolu ✅

Le dashboard Grafana affichait des graphiques vides pour deux raisons :

### 1. **UID de Datasource Incorrect**
- **Ancien**: `"uid": "PROMETHEUS"` (hardcodé)
- **Nouveau**: `"uid": "${DS_PROMETHEUS}"` (variable Grafana auto-remplacée)
- ✅ **Corrigé** dans `monitoring/grafana/dashboards/agentic_rag_dashboard.json`

### 2. **Serveur de Test Bloquant le Port 8000**
- Un processus de test occupait le port 8000
- Les vraies requêtes ne pouvaient pas exporter leurs métriques
- ✅ **Résolu** : Processus arrêté avec `kill`

## Utilisation du Monitoring

### Démarrer la Stack Monitoring
```bash
docker compose -f monitoring/docker-compose.yml up -d
```

### Arrêter un Serveur de Test sur Port 8000 (si nécessaire)
```bash
# Trouver le processus
ps aux | grep "metrics_server" | grep -v grep

# Arrêter le processus
pkill -f "metrics_server"
# ou
kill <PID>
```

### Exécuter une Requête pour Générer des Métriques
```bash
source .venv/bin/activate
python -m agentic_rag.cli query --prompt "What is the main contribution of paper 2407.14477?"
```

### Accéder aux Interfaces

#### Grafana Dashboard
- **URL**: http://localhost:3000
- **Login**: `admin` / `admin`
- **Dashboard**: "Agentic RAG Observability" (auto-provisionné)

#### Prometheus
- **URL**: http://localhost:9090
- **Explorer**: http://localhost:9090/graph

#### Métriques Brutes
- **URL**: http://localhost:8000/metrics

## Métriques Disponibles

### LLM Metrics
- `llm_latency_seconds` - Histogram de latence par modèle
- `llm_requests_total{model, status}` - Compteur de requêtes (success/error)

### Agent Metrics
- `agent_step_latency_seconds` - Latence par étape de la boucle agentique
- `agent_loops_total` - Nombre total d'itérations

### System Metrics
- `process_cpu_seconds` - CPU utilisé par le processus
- Métriques Python standard (GC, mémoire)

## Dashboard Grafana - Panneaux

1. **LLM p95 Latency** - Percentile 95 de latence LLM
2. **Agent Loop Avg Latency** - Latence moyenne des étapes agent
3. **Total LLM Requests** - Stat: Total de requêtes
4. **Agent Steps** - Stat: Nombre d'étapes
5. **LLM Request Rate** - Taux de requêtes par minute
6. **LLM Latency (All Calls)** - Latence moyenne globale
7. **LLM Error Rate** - Taux d'erreurs LLM

## Requêtes Prometheus Utiles

```promql
# Latence moyenne LLM
llm_latency_seconds_sum / llm_latency_seconds_count

# Nombre total de requêtes LLM
sum(llm_requests_total)

# Taux d'erreur LLM
rate(llm_requests_total{status="error"}[5m])

# Agent loops par minute
rate(agent_loops_total[1m])
```

## Troubleshooting

### Dashboard vide
1. Vérifier que l'app est lancée et exporte des métriques: `curl http://localhost:8000/metrics`
2. Vérifier que Prometheus scrape: `curl http://localhost:9090/api/v1/targets`
3. Tester une requête dans Prometheus Explorer
4. Vérifier le datasource Grafana: Settings > Data Sources

### Port 8000 déjà occupé
```bash
# Trouver et arrêter le processus
lsof -ti:8000 | xargs kill -9
```

### Grafana ne démarre pas
```bash
docker compose -f monitoring/docker-compose.yml down
docker compose -f monitoring/docker-compose.yml up -d
docker logs -f agentic_rag_grafana
```

## Architecture Monitoring

```
Application (port 8000)
    ↓ expose /metrics
Prometheus (port 9090)
    ↓ scrape every 15s
    ↓ datasource
Grafana (port 3000)
    ↓ visualize
Dashboard "Agentic RAG Observability"
```

## Auto-provisioning Grafana

Les fichiers suivants sont automatiquement chargés au démarrage :
- `monitoring/grafana/provisioning/datasources/datasource.yml` - Datasource Prometheus
- `monitoring/grafana/provisioning/dashboards/dashboard.yml` - Configuration de provisioning
- `monitoring/grafana/dashboards/agentic_rag_dashboard.json` - Dashboard principal

## Next Steps

Pour générer un flux constant de métriques :
```bash
# Lancer une évaluation complète
python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment monitoring_test
```

Cela générera de nombreuses requêtes et remplira vos graphiques ! 🎉
