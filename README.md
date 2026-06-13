# Agentic RAG for arXiv

Assistant de recherche scientifique basé sur un pipeline **Agentic RAG** pour articles arXiv.

Le projet permet de chercher des papiers, télécharger leurs PDF, les parser, les indexer localement, poser des questions avec récupération de contexte, évaluer automatiquement les réponses, puis visualiser les résultats dans des interfaces Streamlit.

Son objectif principal est de montrer un RAG **mesurable et améliorable** : on peut créer une baseline, modifier le retrieval, le chunking, le prompt ou le reranking, puis comparer les scores dans un dashboard.

---

## Ce que le Projet Fait

- Recherche et téléchargement d’articles arXiv.
- Parsing de PDF scientifiques avec détection de sections.
- Chunking avec métadonnées de citation et détection de blocs importants (`definition`, `example`, `theorem`, `lemma`, etc.).
- Indexation locale FAISS + BM25.
- Recherche hybride dense/sparse avec boost metadata et reranking optionnel.
- Boucle agentique pilotée par un LLM.
- Génération de réponses à partir du contexte récupéré.
- Évaluation RAG Triad : context relevance, groundedness, answer relevance.
- Runner d’expériences reproductibles.
- Dashboard qualité pour comparer les versions.
- UI simple pour écrire un prompt et voir la réponse.
- Monitoring Prometheus + Grafana pour observer les métriques LLM et agent.

---

## Démarrage Rapide

### Option recommandée : tout lancer avec `run.sh`

Le script `run.sh` lance le flux complet :

```text
setup -> monitoring -> query évaluée -> eval-run -> dashboards Streamlit
```

Première exécution, si la venv n’existe pas encore :

```bash
./run.sh --install
```

Exécution rapide par défaut :

```bash
./run.sh
```

Par défaut, le script lance une évaluation courte avec `--limit 1` et l’expérience `smoke_test`.

Variantes utiles :

```bash
# Évaluation complète
./run.sh --full --experiment baseline

# Évaluation courte avec un nom dédié
./run.sh --limit 3 --experiment monitoring_test

# Lancer sans dashboards Streamlit
./run.sh --no-dashboards

# Lancer sans monitoring Docker
./run.sh --skip-monitoring
```

Services exposés :

```text
RAG Quality Dashboard: http://localhost:8501
Query UI:               http://localhost:8502
Grafana:                http://localhost:3000  admin/admin
Prometheus:             http://localhost:9090
Metrics app:            http://localhost:8000/metrics
```

---

## Démarrage Manuel

### 1. Installer l’environnement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurer le projet

```bash
cp .env.example .env
```

Par défaut, utilisez `LLM_PROVIDER=mock` pour vérifier le flux sans clé API. Pour de vraies réponses, configurez `ollama`, `gemini` ou `openai` dans `.env`.

### 3. Tester la CLI

```bash
.venv/bin/python -m agentic_rag.cli --help
```

### 4. Poser une question

```bash
.venv/bin/python -m agentic_rag.cli query \
  --prompt "What is the main contribution of paper 2407.14477?"
```

### 5. Ouvrir l’UI interactive

```bash
.venv/bin/streamlit run dashboard/query_ui.py
```

### 6. Lancer une expérience d’évaluation

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

### 7. Ouvrir le dashboard qualité

```bash
.venv/bin/streamlit run dashboard/app.py
```

### 8. Démarrer le monitoring Grafana/Prometheus

```bash
docker compose -f monitoring/docker-compose.yml up -d
```

Si votre machine utilise l’ancien Compose :

```bash
docker-compose -f monitoring/docker-compose.yml up -d
```

Le monitoring lit les métriques exposées par l’application sur :

```text
http://localhost:8000/metrics
```

Ce serveur de métriques démarre quand l’orchestrateur Python tourne, par exemple pendant `query`, `eval-run` ou l’UI Streamlit.

---

## Structure Courte

```text
agentic_rag/       # Pipeline RAG, agents, retrieval, évaluation
dashboard/         # Interfaces Streamlit
data/              # PDF, index vectoriel, dataset d’évaluation
docs/              # Documentation détaillée
runs/              # Résultats d’expériences, ignorés par Git
monitoring/        # Prometheus, Grafana, alerting rules
```

---

## Documentation Complète

La documentation détaillée se trouve dans [docs/](docs/README.md).

- [Installation et configuration](docs/setup.md)
- [Architecture interne](docs/architecture.md)
- [Utilisation CLI](docs/cli.md)
- [Interfaces Streamlit](docs/interfaces.md)
- [Évaluation et expériences](docs/evaluation.md)
- [Données locales et index](docs/data.md)
- [Améliorer la qualité du RAG](docs/improvement.md)
- [Dépannage](docs/troubleshooting.md)
- [Déploiement AWS staging](docs/aws_deployment.md)
- [Guide de monitoring](MONITORING_GUIDE.md)

---

## Exemple de Workflow

```bash
# Réingérer un papier si le parser/chunking a changé
.venv/bin/python -m agentic_rag.cli ingest --ids "2407.14477"

# Créer une baseline
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline

# Modifier un paramètre dans .env ou dans le code

# Lancer une version améliorée
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1

# Comparer les résultats
.venv/bin/streamlit run dashboard/app.py
```

---

## Dashboards

### RAG Quality Dashboard

```bash
.venv/bin/streamlit run dashboard/app.py
```

Ce dashboard lit les artefacts sauvegardés dans `runs/`. Il ne reflète pas automatiquement les changements de code : il faut relancer une évaluation pour créer un nouveau dossier d’expérience.

Exemple :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment block_aware_v1 \
  --limit 1
```

### Grafana Observability

```bash
docker compose -f monitoring/docker-compose.yml up -d
```

Puis ouvrir :

```text
http://localhost:3000
```

Login par défaut :

```text
admin / admin
```

Le dashboard principal est :

```text
Agentic RAG Observability
```

Il expose notamment :

```text
LLM latency/request rate
Agent loop latency
Retrieval p95 latency by mode
Retrieval request rate by mode/status
Returned chunks average by mode
Reranker p95 latency
```

Prometheus est disponible sur :

```text
http://localhost:9090
```

---

## Chunking Block-Aware

Le parser PDF détecte maintenant des blocs scientifiques importants dans chaque section :

```text
definition, example, theorem, proposition, lemma, remark, corollary, notation
```

Chaque chunk peut contenir :

```json
{
  "section": "Preliminaries",
  "block_type": "definition",
  "block_index": 0,
  "metadata_boost": 0.15
}
```

Pour les questions de définition, le retrieval favorise les chunks `definition` ainsi que les sections comme `Abstract`, `Introduction`, `Background`, `Preliminaries`, `Definitions` et `Examples`.

Si le parser ou le chunking change, réingérez les papiers pour mettre à jour `data/vector_store/` :

```bash
.venv/bin/python -m agentic_rag.cli ingest --ids "2407.14477"
```

---

## Comparer les Modes de Retrieval

Le retrieval peut être forcé avec `RETRIEVAL_MODE` ou `--retrieval-mode` :

```text
faiss
bm25
hybrid
hybrid_reranker
```

Exemple :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment hybrid_reranker_v1 \
  --retrieval-mode hybrid_reranker \
  --limit 1
```

Cela prépare l’étude comparative :

```text
FAISS seul -> BM25 seul -> Hybrid -> Hybrid + Reranker
```

Commande benchmark tout-en-un :

```bash
.venv/bin/python -m agentic_rag.cli benchmark \
  --dataset data/eval/questions.jsonl \
  --experiment retrieval_study_v1 \
  --limit 1
```

Cette commande crée :

```text
runs/retrieval_study_v1/comparison.json
runs/retrieval_study_v1_faiss/
runs/retrieval_study_v1_bm25/
runs/retrieval_study_v1_hybrid/
runs/retrieval_study_v1_hybrid_reranker/
```

Pendant l'exécution, Prometheus collecte aussi les métriques `retrieval_latency_seconds`,
`retrieval_requests_total`, `retrieved_chunks_count` et `reranker_latency_seconds`.
Grafana permet donc de comparer les modes sur la qualité dans `runs/` et sur le coût
d'exécution dans le dashboard observability.

---

## Données Déjà Présentes

Le projet contient déjà un article arXiv indexé pour faciliter les premiers tests :

```text
2407.14477
Data-Centric Human Preference with Rationales for Direct Preference Alignment
```

Fichiers associés :

```text
data/pdfs/2407.14477.pdf
data/vector_store/index.faiss
data/vector_store/metadata.pkl
```

---

## Commandes Utiles

```bash
# Recherche arXiv
.venv/bin/python -m agentic_rag.cli search --query "alignment with rationales" --limit 5

# Ingestion
.venv/bin/python -m agentic_rag.cli ingest --ids "2407.14477"

# Question RAG avec évaluation
.venv/bin/python -m agentic_rag.cli query --prompt "What are the paper limitations?" --evaluate

# UI prompt
.venv/bin/streamlit run dashboard/query_ui.py

# Dashboard qualité
.venv/bin/streamlit run dashboard/app.py

# Monitoring
docker compose -f monitoring/docker-compose.yml up -d

# Tout lancer
./run.sh
```

---

## Résumé

Ce dépôt contient un système complet :

```text
arXiv search -> PDF parsing -> chunking -> hybrid retrieval -> agentic reasoning -> answer -> evaluation -> dashboard
```

Le but est de construire un assistant scientifique et de prouver l’amélioration de la qualité des réponses avec des expériences reproductibles.
