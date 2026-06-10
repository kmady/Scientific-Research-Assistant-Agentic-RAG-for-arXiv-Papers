# Données Locales et Index

Cette page décrit les dossiers de données du projet.

## Structure

```text
data/
  eval/
    questions.jsonl
  pdfs/
    2407.14477.pdf
  vector_store/
    index.faiss
    metadata.pkl
```

## `data/eval/`

Contient les datasets d’évaluation.

Fichier principal :

```text
data/eval/questions.jsonl
```

Chaque ligne est un cas d’évaluation indépendant.

## `data/pdfs/`

Contient les PDF téléchargés depuis arXiv.

Le projet inclut déjà :

```text
2407.14477.pdf
```

Article :

```text
Data-Centric Human Preference with Rationales for Direct Preference Alignment
```

## `data/vector_store/`

Contient l’index local :

```text
index.faiss
metadata.pkl
```

`index.faiss` stocke l’index vectoriel dense.

`metadata.pkl` stocke :

- les chunks ;
- les métadonnées ;
- les embeddings.

## Ajouter des Documents

Par ID :

```bash
.venv/bin/python -m agentic_rag.cli ingest --ids "2305.18290"
```

Par recherche :

```bash
.venv/bin/python -m agentic_rag.cli ingest --query "agentic rag" --limit 5
```

## Éviter les Doublons

`VectorStore.add_chunks(...)` filtre les chunks déjà présents avec la clé :

```text
(arxiv_id, section, chunk_index)
```

Cela évite de réindexer exactement les mêmes chunks.

## Données Versionnées ou Non

Actuellement, certains fichiers de données sont présents dans le dépôt pour faciliter la démonstration.

Le dossier `runs/` est ignoré par Git, car les résultats d’expériences peuvent être volumineux et dépendre de la machine, du provider LLM et du moment d’exécution.

Le fichier `.env` est aussi ignoré par Git.
