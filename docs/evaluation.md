# Évaluation et Expériences

Cette page explique comment le projet mesure la qualité des réponses et comment comparer plusieurs versions du RAG.

## RAG Triad

Le fichier `agentic_rag/evaluator.py` implémente trois métriques principales.

| Métrique | Question évaluée |
|---|---|
| Context Relevance | Le contexte récupéré aide-t-il vraiment à répondre ? |
| Groundedness | La réponse est-elle supportée par le contexte ? |
| Answer Relevance | La réponse répond-elle bien à la question ? |

Le score global est la moyenne des trois scores.

Le backend par défaut est :

```text
llm_judge
```

Il se configure avec `EVALUATION_BACKEND` ou `--evaluation-backend`.
Les backends `ragas` et `deepeval` sont réservés comme intégrations optionnelles
pour la suite de l'étude comparative.

## LLM-as-a-Judge

L’évaluateur envoie au LLM :

- la question utilisateur ;
- les chunks récupérés ;
- la réponse générée.

Le LLM doit répondre en JSON :

```json
{
  "score": 0.85,
  "reason": "Brief explanation of the score."
}
```

## Dataset d’Évaluation

Le dataset est dans :

```text
data/eval/questions.jsonl
```

Format d’une ligne :

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

- `id` : identifiant stable ;
- `question` : prompt à poser ;
- `paper_ids` : documents attendus ;
- `expected_topics` : sujets attendus dans une bonne réponse ;
- `question_type` : type de question.

## Lancer une Expérience

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline \
  --evaluation-backend llm_judge
```

Limiter à une question :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment smoke_test \
  --evaluation-backend llm_judge \
  --limit 1
```

## Artefacts Générés

Chaque run crée :

```text
runs/<experiment>/
  config.json
  results.jsonl
  summary.json
```

### `config.json`

Snapshot de configuration :

- provider LLM ;
- modèle Ollama ;
- provider embeddings ;
- modèle d’embeddings ;
- poids BM25 / dense ;
- paramètres de chunking ;
- statut du reranker ;
- backend d'évaluation.

### `results.jsonl`

Une ligne par question :

- question ;
- réponse ;
- étapes agentiques ;
- chunks récupérés ;
- scores ;
- latence ;
- métadonnées attendues.

### `summary.json`

Scores moyens :

- `overall_rag_score`
- `avg_context_relevance`
- `avg_groundedness`
- `avg_answer_relevance`
- `avg_latency_seconds`
- `evaluation_backend`

## RAGAS et DeepEval

Le code accepte déjà :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment ragas_probe \
  --evaluation-backend ragas \
  --limit 1
```

et :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment deepeval_probe \
  --evaluation-backend deepeval \
  --limit 1
```

Installer les dépendances optionnelles :

```bash
.venv/bin/python -m pip install -r requirements-eval.txt
```

Les scores sont normalisés vers les champs communs :

| Champ projet | RAGAS | DeepEval |
|---|---|---|
| `context_relevance` | `context_precision` | `ContextualRelevancyMetric` |
| `groundedness` | `faithfulness` | `FaithfulnessMetric` |
| `answer_relevance` | `answer_relevancy` | `AnswerRelevancyMetric` |

Si les dépendances ou la configuration fournisseur manquent, le résultat indique
explicitement l'erreur dans `backend_error` ou dans la raison de la métrique.

## Comparer Deux Versions

Workflow typique :

```bash
# Version actuelle
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline

# Modifier le système
# Exemple : changer BM25_WEIGHT, prompt, chunking ou reranker.

# Nouvelle version
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1

# Comparaison visuelle
.venv/bin/streamlit run dashboard/app.py
```

## Limites de l’Évaluation Actuelle

- Le jugement dépend du LLM utilisé.
- Les métriques ne vérifient pas encore explicitement l’exactitude des citations.
- Le dataset initial est petit.
- `expected_topics` est stocké, mais pas encore utilisé comme score automatique dédié.

## Métriques Futures

Métriques utiles à ajouter :

- citation accuracy ;
- retrieval recall ;
- answer completeness ;
- hallucination rate ;
- coût par question ;
- tokens consommés ;
- comparaison automatique baseline vs improved.
