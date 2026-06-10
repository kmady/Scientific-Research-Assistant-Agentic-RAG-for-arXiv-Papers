# Interfaces Streamlit

Le projet fournit deux interfaces web Streamlit :

- une UI simple pour poser des prompts ;
- un dashboard pour comparer les expériences d’évaluation.

## UI Interactive de Prompt

Commande :

```bash
.venv/bin/streamlit run dashboard/query_ui.py
```

Fonctionnalités :

- écrire une question ;
- lancer le pipeline Agentic RAG ;
- voir la réponse finale ;
- inspecter les étapes de l’agent ;
- afficher les chunks récupérés ;
- cocher `Evaluate` pour évaluer la réponse.

Quand l’utiliser :

- tester rapidement une question ;
- explorer les documents indexés ;
- vérifier si l’agent récupère les bons passages ;
- démontrer le projet de façon interactive.

## Dashboard Qualité

Commande :

```bash
.venv/bin/streamlit run dashboard/app.py
```

Le dashboard lit les dossiers dans :

```text
runs/
```

Il affiche :

- score global par expérience ;
- scores context relevance, groundedness, answer relevance ;
- scores question par question ;
- réponses générées ;
- chunks récupérés ;
- étapes de l’agent ;
- configuration de chaque expérience.

## Workflow Dashboard

Créer une baseline :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

Modifier une partie du système, puis créer une nouvelle expérience :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1
```

Ouvrir le dashboard :

```bash
.venv/bin/streamlit run dashboard/app.py
```

## État Vide

Si aucun dossier `runs/<experiment>/` n’existe, le dashboard affiche un message indiquant comment créer une baseline.

## Conseils d’Usage

- Utiliser des noms d’expérience explicites : `baseline`, `reranker_disabled`, `chunk_1500`, `prompt_v2`.
- Garder une baseline stable pour les comparaisons.
- Lancer une expérience courte avec `--limit 1` avant une expérience complète.
- Ne pas versionner `runs/`, car ces résultats sont propres à une machine et peuvent devenir volumineux.
