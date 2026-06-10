# Dépannage

## `ModuleNotFoundError: No module named 'fitz'`

PyMuPDF n’est pas installé dans l’environnement actif.

Solution :

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Vérifier :

```bash
.venv/bin/python - <<'PY'
import fitz
print("fitz ok")
PY
```

## `ModuleNotFoundError: No module named 'dashboard'`

Lancer Streamlit depuis la racine du projet.

Pour le dashboard qualité :

```bash
.venv/bin/streamlit run dashboard/app.py
```

Pour l’UI de prompt :

```bash
.venv/bin/streamlit run dashboard/query_ui.py
```

## Le Dashboard est Vide

Aucune expérience n’a encore été générée.

Créer une baseline :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

Puis relancer :

```bash
.venv/bin/streamlit run dashboard/app.py
```

## Ollama ne Répond Pas

Vérifier que le serveur tourne :

```bash
ollama serve
```

Vérifier les modèles installés :

```bash
ollama list
```

Installer le modèle attendu :

```bash
ollama pull qwen2.5:7b
```

Vérifier `.env` :

```env
LLM_PROVIDER=ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

## Le Premier Retrieval est Lent

Les modèles locaux d’embeddings et de reranking peuvent être téléchargés au premier usage.

Modèles par défaut :

```text
BAAI/bge-large-en-v1.5
BAAI/bge-reranker-base
```

## arXiv ne Télécharge Pas le PDF

Causes possibles :

- problème réseau ;
- rate limit ;
- lien PDF invalide ;
- arXiv renvoie une page HTML au lieu du PDF.

Le code détecte certains cas HTML et retourne `None`.

Réessayer plus tard ou vérifier le lien PDF manuellement.

## Erreur JSON dans la Boucle Agentique

L’orchestrateur attend une réponse JSON du LLM. S’il reçoit une réponse mal formée, il ajoute une observation d’erreur et laisse le modèle se corriger au tour suivant.

Si cela arrive souvent :

- réduire la température ;
- utiliser un modèle plus fiable en JSON ;
- renforcer le system prompt ;
- utiliser le mode JSON du provider si disponible.

## Ne Pas Pousser `.env`

Vérifier avant commit :

```bash
git status --short
```

`.env` doit être ignoré.

## Vérifier les Dépendances

```bash
.venv/bin/python -m pip check
```

## Vérifier la Syntaxe Python

```bash
.venv/bin/python -m py_compile \
  agentic_rag/cli.py \
  agentic_rag/experiments.py \
  dashboard/app.py \
  dashboard/query_ui.py \
  dashboard/loaders.py
```
