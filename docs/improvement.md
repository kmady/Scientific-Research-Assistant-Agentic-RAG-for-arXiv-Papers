# Améliorer la Qualité du RAG

Le projet est conçu pour mesurer l’impact des changements. Chaque amélioration devrait être testée avec une nouvelle expérience et comparée à une baseline.

## Workflow

```bash
# 1. Baseline
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline

# 2. Modifier une partie du système

# 3. Nouvelle expérience
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1

# 4. Comparer
.venv/bin/streamlit run dashboard/app.py
```

## Axes d’Amélioration

### Chunking

Paramètres :

- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

Hypothèses à tester :

- chunks plus grands pour conserver plus de contexte ;
- chunks plus petits pour améliorer la précision du retrieval ;
- overlap plus élevé pour éviter de couper les idées importantes.

### Recherche Hybride

Paramètres :

- `BM25_WEIGHT`
- `DENSE_WEIGHT`

Exemples d’expériences :

```text
hybrid_30_70
hybrid_50_50
hybrid_70_30
```

### Reranking

Paramètres :

- `USE_RERANKER`
- `LOCAL_RERANKER_MODEL`

Comparer :

```text
reranker_enabled
reranker_disabled
```

### Prompting

Améliorations possibles :

- demander explicitement des citations ;
- demander une réponse structurée ;
- forcer le refus si le contexte est insuffisant ;
- séparer comparaison, limites et résultats.

### Query Rewriting

Ajouter une étape qui transforme la question utilisateur en requêtes de retrieval plus ciblées.

Exemple :

```text
Question utilisateur:
How does the paper use rationales to improve preference alignment?

Requêtes retrieval:
- rationales preference alignment method
- rationale-augmented preference dataset
- direct preference alignment rationales experiments
```

### Section-Aware Retrieval

Utiliser les sections détectées pour favoriser certains passages :

- `Method` pour les questions de méthode ;
- `Experiments` pour les résultats ;
- `Limitations` pour les limites ;
- `Abstract` et `Introduction` pour les résumés.

### Retrieval Modes

Les expériences peuvent maintenant forcer un mode retrieval :

```text
faiss
bm25
hybrid
hybrid_reranker
```

Cela permet de comparer proprement :

```bash
python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment faiss_v1 --retrieval-mode faiss
python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment bm25_v1 --retrieval-mode bm25
python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment hybrid_v1 --retrieval-mode hybrid
python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment hybrid_reranker_v1 --retrieval-mode hybrid_reranker
```

### Block-Aware Retrieval

Le parser extrait aussi les blocs importants des articles scientifiques :

```text
definition, example, theorem, proposition, lemma, remark, corollary, notation
```

Chaque chunk peut contenir :

```json
{
  "block_type": "definition",
  "metadata_boost": 0.15
}
```

Pour une question de définition, le retrieval favorise les chunks `definition` ainsi que les sections comme `Abstract`, `Introduction`, `Background`, `Preliminaries`, `Definitions` et `Examples`.

## Nommer les Expériences

Utiliser des noms explicites :

```text
baseline
prompt_v2
chunk_1500_overlap_150
reranker_off
hybrid_dense_050
section_retrieval_v1
```

## Lire les Résultats

Le dashboard montre :

- score global ;
- groundedness ;
- context relevance ;
- answer relevance ;
- latence ;
- détails par question.

Une amélioration utile devrait augmenter les scores sans faire exploser la latence.

## Prochaines Métriques à Ajouter

- citation accuracy ;
- retrieval recall ;
- answer completeness ;
- hallucination rate ;
- coût ;
- consommation tokens.
