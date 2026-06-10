# Utilisation CLI

Toutes les commandes se lancent depuis la racine du projet. Les exemples utilisent l’environnement virtuel local.

## Aide

```bash
.venv/bin/python -m agentic_rag.cli --help
```

## Chercher des Articles arXiv

```bash
.venv/bin/python -m agentic_rag.cli search \
  --query "Direct Preference Optimization" \
  --limit 5
```

Cette commande affiche :

- ID arXiv ;
- titre ;
- auteurs ;
- date de publication.

## Ingérer des Articles par ID

```bash
.venv/bin/python -m agentic_rag.cli ingest \
  --ids "2305.18290,2402.01306"
```

Étapes réalisées :

1. récupération des métadonnées arXiv ;
2. téléchargement des PDF ;
3. parsing des sections ;
4. chunking ;
5. embeddings ;
6. sauvegarde dans le vector store.

## Ingérer des Articles par Requête

```bash
.venv/bin/python -m agentic_rag.cli ingest \
  --query "agentic rag" \
  --limit 5
```

Le système cherche les articles correspondants et tente d’ingérer les `limit` premiers résultats.

## Poser une Question

```bash
.venv/bin/python -m agentic_rag.cli query \
  --prompt "What is the main contribution of paper 2407.14477?"
```

La sortie contient :

- les étapes de l’agent ;
- les actions choisies ;
- la réponse synthétisée.

## Poser une Question avec Évaluation

```bash
.venv/bin/python -m agentic_rag.cli query \
  --prompt "How does the paper use rationales to improve preference alignment?" \
  --evaluate
```

L’option `--evaluate` lance ensuite `RAGEvaluator` sur la réponse.

## Lancer une Expérience

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

Options :

| Option | Description |
|---|---|
| `--dataset` | Chemin du dataset JSONL. |
| `--experiment` | Nom du dossier dans `runs/`. |
| `--limit` | Nombre maximal de questions à exécuter. |

Exemple rapide sur une seule question :

```bash
.venv/bin/python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment smoke_test \
  --limit 1
```

## Commandes Fréquentes

```bash
# Recherche
.venv/bin/python -m agentic_rag.cli search --query "alignment with rationales" --limit 5

# Ingestion
.venv/bin/python -m agentic_rag.cli ingest --ids "2407.14477"

# Question simple
.venv/bin/python -m agentic_rag.cli query --prompt "Summarize paper 2407.14477."

# Question évaluée
.venv/bin/python -m agentic_rag.cli query --prompt "What are the paper limitations?" --evaluate

# Expérience
.venv/bin/python -m agentic_rag.cli eval-run --dataset data/eval/questions.jsonl --experiment baseline
```
