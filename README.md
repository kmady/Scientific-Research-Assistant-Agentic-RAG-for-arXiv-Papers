# Agentic RAG for arXiv

Assistant de recherche scientifique basé sur un pipeline **Agentic RAG** pour articles arXiv.

Le projet permet de chercher des papiers, télécharger leurs PDF, les parser, les indexer localement, poser des questions avec récupération de contexte, évaluer automatiquement les réponses, puis visualiser les résultats dans des interfaces Streamlit.

Son objectif principal est de montrer un RAG **mesurable et améliorable** : on peut créer une baseline, modifier le retrieval, le chunking, le prompt ou le reranking, puis comparer les scores dans un dashboard.

---

## Ce que le Projet Fait

- Recherche et téléchargement d’articles arXiv.
- Parsing de PDF scientifiques avec détection de sections.
- Chunking avec métadonnées de citation.
- Indexation locale FAISS + BM25.
- Recherche hybride dense/sparse avec reranking optionnel.
- Boucle agentique pilotée par un LLM.
- Génération de réponses à partir du contexte récupéré.
- Évaluation RAG Triad : context relevance, groundedness, answer relevance.
- Runner d’expériences reproductibles.
- Dashboard qualité pour comparer les versions.
- UI simple pour écrire un prompt et voir la réponse.

---

## Démarrage Rapide

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

---

## Structure Courte

```text
agentic_rag/       # Pipeline RAG, agents, retrieval, évaluation
dashboard/         # Interfaces Streamlit
data/              # PDF, index vectoriel, dataset d’évaluation
docs/              # Documentation détaillée
runs/              # Résultats d’expériences, ignorés par Git
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

---

## Exemple de Workflow

```bash
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
```

---

## Résumé

Ce dépôt contient un système complet :

```text
arXiv search -> PDF parsing -> chunking -> hybrid retrieval -> agentic reasoning -> answer -> evaluation -> dashboard
```

Le but est de construire un assistant scientifique et de prouver l’amélioration de la qualité des réponses avec des expériences reproductibles.
