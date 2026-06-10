# Agentic RAG for arXiv: Scientific Research Assistant

Assistant de recherche scientifique basé sur un pipeline **Agentic RAG** pour articles arXiv. Le projet permet de chercher des papiers, télécharger leurs PDF, les parser, les indexer localement, interroger les documents avec un LLM, évaluer automatiquement la qualité des réponses, puis visualiser les résultats dans des interfaces Streamlit.

L’objectif principal n’est pas seulement de produire des réponses, mais de construire un système RAG **mesurable et améliorable** : chaque modification du retrieval, du chunking, du prompt ou du reranking peut être comparée à une baseline avec des scores d’évaluation.

---

## Fonctionnalités

- Recherche d’articles via l’API arXiv.
- Téléchargement automatique des PDF.
- Parsing de PDF scientifiques avec détection approximative des sections.
- Découpage en chunks avec métadonnées : titre, auteurs, arXiv ID, section, pages.
- Indexation locale avec FAISS.
- Recherche hybride :
  - recherche dense par embeddings ;
  - recherche sparse BM25 ;
  - fusion des scores ;
  - reranking optionnel.
- Boucle agentique avec actions :
  - `search_arxiv`
  - `download_and_index`
  - `retrieve_context`
  - `answer_user`
- Support LLM :
  - Ollama ;
  - Gemini ;
  - OpenAI ;
  - mock mode.
- Évaluation automatique RAG Triad :
  - context relevance ;
  - groundedness / faithfulness ;
  - answer relevance.
- Runner d’expériences reproductibles.
- Dashboard qualité pour comparer les expériences.
- UI simple pour écrire un prompt et voir la réponse.

---

## Structure du Projet

```text
agentic_rag/
  cli.py              # Interface en ligne de commande
  config.py           # Configuration via .env
  evaluator.py        # Évaluation RAG Triad
  experiments.py      # Runner d’expériences reproductibles
  llm.py              # Clients LLM : Ollama, Gemini, OpenAI, Mock
  orchestrator.py     # Boucle agentique RAG
  pdf_parser.py       # Extraction PDF, sections, chunking
  search.py           # Recherche et téléchargement arXiv
  vector_db.py        # FAISS, BM25, embeddings, reranking

dashboard/
  app.py              # Dashboard de comparaison des expériences
  loaders.py          # Chargement des résultats runs/*
  query_ui.py         # UI simple pour poser des questions

data/
  eval/
    questions.jsonl   # Dataset d’évaluation
  pdfs/               # PDF téléchargés
  vector_store/       # Index FAISS et métadonnées

runs/
  baseline/           # Résultats d’expériences, ignorés par Git

requirements.txt
.env.example
.gitignore
README.md
```

---

## Installation

### 1. Créer et activer l’environnement Python

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

Ou sans activer l’environnement :

```bash
.venv/bin/python -m pip install -r requirements.txt
```

### 3. Vérifier l’installation

```bash
.venv/bin/python -m pip check
.venv/bin/python -m agentic_rag.cli --help
```

---

## Configuration

Copier le fichier d’exemple :

```bash
cp .env.example .env
```

Le fichier `.env` est ignoré par Git. Il peut contenir des clés API et ne doit pas être poussé dans le dépôt.

Exemple :

```env
LLM_PROVIDER=mock

GEMINI_API_KEY=
OPENAI_API_KEY=

OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIM=1024

USE_RERANKER=true
LOCAL_RERANKER_MODEL=BAAI/bge-reranker-base

BM25_WEIGHT=0.3
DENSE_WEIGHT=0.7
RETRIEVAL_TOP_K=30
RERANK_TOP_N=8
```

Notes importantes :

- `.env.example` utilise `LLM_PROVIDER=mock` pour tester sans clé API.
- `config.py` retombe sur `ollama` si aucune variable `LLM_PROVIDER` n’est fournie.
- Les modèles locaux d’embeddings et de reranking sont téléchargés via `sentence-transformers` au premier usage.
- Pour Ollama, le serveur local doit tourner avant les requêtes :

```bash
ollama serve
ollama pull qwen2.5:7b
```

---

## Utilisation CLI

Toutes les commandes se lancent depuis la racine du projet.

### Chercher des articles arXiv

```bash
.venv/bin/python -m agentic_rag.cli search \
  --query "Direct Preference Optimization" \
  --limit 5
```

Cette commande interroge l’API arXiv et affiche les articles correspondants.

### Ingérer des articles par ID

```bash
.venv/bin/python -m agentic_rag.cli ingest \
  --ids "2305.18290,2402.01306"
```

Cette commande :

1. récupère les métadonnées arXiv ;
2. télécharge les PDF ;
3. parse les documents ;
4. découpe les sections en chunks ;
5. ajoute les chunks au vector store local.

### Ingérer des articles par recherche

```bash
.venv/bin/python -m agentic_rag.cli ingest \
  --query "agentic rag" \
  --limit 5
```

### Poser une question au RAG

```bash
.venv/bin/python -m agentic_rag.cli query \
  --prompt "What is the main contribution of paper 2407.14477?"
```

### Poser une question avec évaluation automatique

```bash
.venv/bin/python -m agentic_rag.cli query \
  --prompt "How does the paper use rationales to improve preference alignment?" \
  --evaluate
```

---

## UI Interactive de Prompt

Pour poser une question depuis une interface web simple :

```bash
.venv/bin/streamlit run dashboard/query_ui.py
```

Cette UI permet de :

- écrire un prompt ;
- lancer la boucle Agentic RAG ;
- afficher la réponse finale ;
- inspecter les étapes de l’agent ;
- afficher les chunks récupérés ;
- cocher `Evaluate` pour scorer la réponse.

---

## Évaluation RAG

Le projet contient un évaluateur dans `agentic_rag/evaluator.py`. Il utilise un LLM-as-a-Judge pour produire trois scores :

| Métrique | Description |
|---|---|
| Context Relevance | Les chunks récupérés sont-ils utiles pour répondre à la question ? |
| Groundedness | Les affirmations de la réponse sont-elles supportées par le contexte ? |
| Answer Relevance | La réponse traite-t-elle correctement la question ? |

Le score global est la moyenne de ces trois métriques.

---

## Dataset d’Évaluation

Le dataset initial est dans :

```text
data/eval/questions.jsonl
```

Chaque ligne est un objet JSON :

```json
{
  "id": "q001",
  "question": "What is the main contribution of the paper Data-Centric Human Preference with Rationales for Direct Preference Alignment?",
  "paper_ids": ["2407.14477"],
  "expected_topics": ["human preference", "rationales", "direct preference alignment", "data-centric"],
  "question_type": "contribution"
}
```

Champs :

- `id` : identifiant stable de la question ;
- `question` : prompt à poser au RAG ;
- `paper_ids` : articles supposés utiles ;
- `expected_topics` : sujets que la réponse devrait couvrir ;
- `question_type` : catégorie d’analyse.

---

## Lancer une Expérience Reproductible

Pour créer une baseline :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

Pour tester une version améliorée :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1
```

Chaque expérience crée :

```text
runs/<experiment>/
  config.json      # Configuration utilisée
  results.jsonl    # Résultats question par question
  summary.json     # Scores moyens
```

Le dossier `runs/` est ignoré par Git afin d’éviter de versionner des résultats volumineux ou spécifiques à une machine.

---

## Dashboard Qualité

Après avoir lancé au moins une expérience :

```bash
.venv/bin/streamlit run dashboard/app.py
```

Le dashboard lit tous les dossiers dans `runs/` et affiche :

- score global par expérience ;
- comparaison des métriques ;
- scores question par question ;
- réponses générées ;
- chunks récupérés ;
- étapes de l’agent ;
- configuration de chaque expérience.

Workflow recommandé :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline

# Modifier le retrieval, le prompt, le chunking ou le reranking.

.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1

.venv/bin/streamlit run dashboard/app.py
```

---

## Architecture Interne

### 1. Recherche arXiv

Fichier : `agentic_rag/search.py`

Responsabilités :

- interroger l’API arXiv ;
- nettoyer les IDs arXiv ;
- parser les flux Atom XML ;
- télécharger les PDF.

### 2. Parsing PDF

Fichier : `agentic_rag/pdf_parser.py`

Responsabilités :

- extraire les blocs de texte avec PyMuPDF ;
- détecter les titres de section avec taille de police, style et motifs textuels ;
- regrouper le texte par section ;
- produire des chunks avec overlap ;
- conserver les métadonnées de citation.

### 3. Vector Store

Fichier : `agentic_rag/vector_db.py`

Responsabilités :

- calculer les embeddings ;
- stocker les vecteurs avec FAISS ;
- construire un index BM25 ;
- fusionner scores denses et sparse ;
- appliquer un reranker local optionnel ;
- sauvegarder l’index dans `data/vector_store/`.

### 4. LLM Client

Fichier : `agentic_rag/llm.py`

Responsabilités :

- fournir une interface commune `chat(...)` ;
- gérer Ollama, Gemini, OpenAI et Mock ;
- retourner une réponse standardisée.

### 5. Orchestrateur Agentique

Fichier : `agentic_rag/orchestrator.py`

Responsabilités :

- maintenir l’état de la boucle agentique ;
- appeler le LLM avec le system prompt ;
- parser les actions JSON du modèle ;
- exécuter les outils disponibles ;
- retourner la réponse finale et les étapes.

Actions disponibles :

```text
search_arxiv
download_and_index
retrieve_context
answer_user
```

### 6. Évaluateur

Fichier : `agentic_rag/evaluator.py`

Responsabilités :

- formater le contexte récupéré ;
- demander à un LLM juge de scorer la réponse ;
- produire un rapport Markdown ;
- calculer le score global.

### 7. Runner d’Expériences

Fichier : `agentic_rag/experiments.py`

Responsabilités :

- lire un dataset JSONL ;
- exécuter le RAG pour chaque question ;
- relancer une recherche de contexte pour l’évaluation ;
- sauvegarder les résultats ;
- produire un résumé global.

---

## Données Locales

Le projet contient déjà un article indexé :

```text
data/pdfs/2407.14477.pdf
data/vector_store/index.faiss
data/vector_store/metadata.pkl
```

Article :

```text
2407.14477
Data-Centric Human Preference with Rationales for Direct Preference Alignment
```

Ces fichiers permettent de tester le retrieval sans devoir ingérer immédiatement un nouvel article.

---

## Améliorer la Qualité du RAG

Exemples d’expériences à comparer :

```text
baseline
improved_prompt
improved_chunking
reranker_disabled
reranker_enabled
hybrid_weight_50_50
multi_query_retrieval
section_aware_retrieval
```

Axes d’amélioration :

- augmenter ou réduire `CHUNK_SIZE` ;
- ajuster `CHUNK_OVERLAP` ;
- modifier `BM25_WEIGHT` et `DENSE_WEIGHT` ;
- activer ou désactiver le reranker ;
- améliorer le system prompt ;
- ajouter une étape de query rewriting ;
- forcer des citations plus strictes ;
- augmenter le dataset d’évaluation.

Après chaque changement, lancer une nouvelle expérience et comparer dans le dashboard.

---

## Dépannage

### `ModuleNotFoundError: No module named 'fitz'`

PyMuPDF n’est pas installé dans l’environnement actif.

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'dashboard'`

Lancer le dashboard depuis la racine du projet :

```bash
.venv/bin/streamlit run dashboard/app.py
```

### Le LLM ne répond pas avec Ollama

Vérifier que le serveur Ollama tourne :

```bash
ollama serve
```

Vérifier que le modèle existe :

```bash
ollama list
```

### Les embeddings prennent du temps au premier lancement

Les modèles `sentence-transformers` sont téléchargés au premier usage. Cela peut prendre plusieurs minutes selon la connexion.

### Le dashboard est vide

Lancer au moins une expérience :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

### Ne pas versionner `.env`

Le fichier `.env` est déjà listé dans `.gitignore`. Vérifier avant commit :

```bash
git status --short
```

---

## Commandes Utiles

```bash
# Aide CLI
.venv/bin/python -m agentic_rag.cli --help

# Recherche arXiv
.venv/bin/python -m agentic_rag.cli search --query "alignment with rationales" --limit 5

# Ingestion
.venv/bin/python -m agentic_rag.cli ingest --ids "2407.14477"

# Question RAG
.venv/bin/python -m agentic_rag.cli query --prompt "Summarize the paper 2407.14477."

# Question RAG avec évaluation
.venv/bin/python -m agentic_rag.cli query --prompt "What are the paper limitations?" --evaluate

# Expérience
.venv/bin/python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment baseline

# Dashboard qualité
.venv/bin/streamlit run dashboard/app.py

# UI prompt
.venv/bin/streamlit run dashboard/query_ui.py
```

---

## Roadmap

- Ajouter des métriques supplémentaires :
  - citation accuracy ;
  - retrieval recall ;
  - answer completeness ;
  - hallucination rate.
- Ajouter une recherche multi-query.
- Ajouter un mode section-aware retrieval.
- Ajouter une comparaison directe baseline vs improved dans la page d’inspection.
- Exporter les rapports en Markdown ou HTML.
- Ajouter des tests automatisés.
- Ajouter un fichier `pyproject.toml` pour packager proprement le projet.

---

## Résumé

Ce projet est un système complet de recherche augmentée pour littérature scientifique :

```text
arXiv search -> PDF parsing -> chunking -> hybrid retrieval -> agentic reasoning -> cited answer -> evaluation -> dashboard
```

Il est conçu pour démontrer une progression mesurable de la qualité des réponses, pas seulement pour générer des réponses ponctuelles.
