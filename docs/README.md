# Documentation du Projet

Ce dossier contient la documentation détaillée du projet **Agentic RAG for arXiv**. Le README racine sert d’introduction rapide ; les pages ci-dessous expliquent chaque partie du système.

## Sommaire

- [Installation et configuration](setup.md)
- [Architecture interne](architecture.md)
- [Utilisation CLI](cli.md)
- [Interfaces Streamlit](interfaces.md)
- [Évaluation et expériences](evaluation.md)
- [Données locales et index](data.md)
- [Améliorer la qualité du RAG](improvement.md)
- [Dépannage](troubleshooting.md)

## Vue d’ensemble

Le projet implémente un assistant de recherche scientifique basé sur un pipeline Agentic RAG :

```text
arXiv search -> PDF parsing -> chunking -> hybrid retrieval -> agentic reasoning -> cited answer -> evaluation -> dashboard
```

Il permet de :

- chercher des articles sur arXiv ;
- télécharger et parser des PDF scientifiques ;
- indexer les documents dans un vector store local ;
- poser des questions avec récupération de contexte ;
- générer des réponses avec un LLM ;
- évaluer automatiquement la qualité des réponses ;
- comparer plusieurs versions du système dans un dashboard.

## Structure Principale

```text
agentic_rag/
  cli.py
  config.py
  evaluator.py
  experiments.py
  llm.py
  orchestrator.py
  pdf_parser.py
  search.py
  vector_db.py

dashboard/
  app.py
  loaders.py
  query_ui.py

data/
  eval/
  pdfs/
  vector_store/

runs/
```

## Workflow Recommandé

1. Installer l’environnement : [setup.md](setup.md)
2. Ingérer ou utiliser les articles déjà indexés : [cli.md](cli.md)
3. Poser des questions via CLI ou UI : [interfaces.md](interfaces.md)
4. Lancer une baseline d’évaluation : [evaluation.md](evaluation.md)
5. Améliorer un composant du RAG : [improvement.md](improvement.md)
6. Comparer les scores dans le dashboard : [interfaces.md](interfaces.md)
