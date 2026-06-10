# Installation et Configuration

Cette page explique comment préparer l’environnement Python, installer les dépendances et configurer les providers LLM/embeddings.

## Prérequis

- Python 3.11 ou 3.12 recommandé.
- Accès réseau pour installer les dépendances et télécharger les modèles locaux au premier usage.
- Optionnel : Ollama si vous voulez utiliser un modèle local.

## Créer l’Environnement

Depuis la racine du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Ou sans activer l’environnement :

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Vérifier l’installation :

```bash
.venv/bin/python -m pip check
.venv/bin/python -m agentic_rag.cli --help
```

## Configuration `.env`

Créer le fichier local :

```bash
cp .env.example .env
```

Le fichier `.env` est ignoré par Git et peut contenir des clés privées.

Exemple de configuration :

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

## Providers LLM

Le projet supporte quatre modes :

| Provider | Usage |
|---|---|
| `mock` | Test sans vrai modèle, utile pour vérifier le flux applicatif. |
| `ollama` | Modèle local servi par Ollama. |
| `gemini` | API Google Gemini. |
| `openai` | API OpenAI. |

Avec Ollama :

```bash
ollama serve
ollama pull qwen2.5:7b
```

Puis dans `.env` :

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:7b
```

## Embeddings et Reranking

Par défaut, les embeddings utilisent :

```env
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

Le reranker utilise :

```env
USE_RERANKER=true
LOCAL_RERANKER_MODEL=BAAI/bge-reranker-base
```

Ces modèles sont téléchargés par `sentence-transformers` au premier usage.

## Fichiers à Ne Pas Versionner

Sont ignorés par `.gitignore` :

- `.env`
- `.venv/`
- `__pycache__/`
- `runs/`
- `data/cache/`

Avant de committer :

```bash
git status --short
```
