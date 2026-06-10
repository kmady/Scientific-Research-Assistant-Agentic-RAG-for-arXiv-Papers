# 🔬 Agentic RAG for arXiv: Scientific Research Assistant

An autonomous, agentic RAG (Retrieval-Augmented Generation) system designed to search, ingest, parse, retrieve, synthesize, and evaluate answers for scientific queries across arXiv papers. 

Developed specifically for processing deep academic literature with high structural chunk integrity and precise citations.

---

## 🚀 Quick Start

### 1. Activate Environment
Activate the Conda environment created for this project:
```bash
conda activate arxiv_rag
```

### 2. Configure Environment
Set up your LLM keys in the `.env` file (copied from `.env.example`).
```bash
cp .env.example .env
# Open .env and add your API keys (GEMINI_API_KEY, OPENAI_API_KEY, or use local Ollama)
```

By default, `LLM_PROVIDER` is set to `mock` for safe testing. Set it to `ollama`, `gemini`, or `openai` to activate real models.

---

## 🛠️ Command-Line Interface (CLI)

Use the entry point script `agentic_rag/cli.py` to execute tasks:

### A. Search arXiv
Search for matching papers on arXiv:
```bash
python -m agentic_rag.cli search --query "Direct Preference Optimization" --limit 5
```

### B. Ingest and Index Papers
Download, parse, and index specific paper IDs into the local FAISS index:
```bash
# Ingest specific IDs (e.g. DPO and KTO papers)
python -m agentic_rag.cli ingest --ids "2305.18290,2402.01306"

# OR search and ingest papers matching a keyword
python -m agentic_rag.cli ingest --query "agentic rag" --limit 5
```

### C. Ask Research Questions (Agentic RAG)
Run the autonomous agent loop to formulate a plan, retrieve details, and compile a comparative synthesis:
```bash
python -m agentic_rag.cli query --prompt "Compare DPO and KTO alignment methods in terms of contributions and results."
```

### D. Run Automated Evaluation
Instruct the RAG pipeline to run the answer through the RAG Triad evaluation metrics (Context Relevance, Groundedness/Faithfulness, and Answer Relevance):
```bash
python -m agentic_rag.cli query --prompt "What is the core idea of Direct Preference Optimization?" --evaluate
```

### E. Run a Reproducible Evaluation Experiment
Run the full evaluation dataset and save artifacts for dashboarding and comparison:
```bash
python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment baseline
```

This creates:
```text
runs/baseline/
  config.json
  results.jsonl
  summary.json
```

Use a new experiment name after changing retrieval, prompting, chunking, or reranking settings:
```bash
python -m agentic_rag.cli eval-run \
  --dataset data/eval/questions.jsonl \
  --experiment improved_v1
```

The dashboard phase will compare these experiment folders to show whether response quality improved.

### F. Launch the Quality Dashboard
After running at least one evaluation experiment, open the Streamlit dashboard:
```bash
streamlit run dashboard/app.py
```

If you are using the project virtual environment directly:
```bash
.venv/bin/streamlit run dashboard/app.py
```

The dashboard reads all experiment folders under `runs/` and shows:
- overall score comparisons;
- context relevance, groundedness, and answer relevance;
- question-level scores;
- generated answers;
- retrieved chunks and agent steps;
- configuration snapshots for each experiment.

### G. Launch the Prompt UI
Use the lightweight prompt interface to ask questions and inspect answers interactively:
```bash
streamlit run dashboard/query_ui.py
```

If you are using the project virtual environment directly:
```bash
.venv/bin/streamlit run dashboard/query_ui.py
```

This UI lets you type a research prompt, run the Agentic RAG pipeline, inspect the final answer, view the agent steps, browse retrieved chunks, and optionally evaluate the answer.

---

## 📐 Architecture & Key Design Decisions

1. **PDF Ingestion & Structure Parsing ([pdf_parser.py](file:///home/dem/workspace/agentic_rag_4_arxiv_AG/agentic_rag/pdf_parser.py))**:
   - Uses PyMuPDF (`fitz`) to extract text blocks.
   - Detects sections (e.g., *Introduction*, *Methodology*, *Experiments*) dynamically by parsing font sizes, styles, and numbering schemas.
   - Preserves section titles and page numbers for **precise citations** matching the format `[ArXiv:XXXX.YYYY, Sec: Section Name, p. Z]`.
2. **Hybrid Search & Reranking ([vector_db.py](file:///home/dem/workspace/agentic_rag_4_arxiv_AG/agentic_rag/vector_db.py))**:
   - **Dense Search**: Computes local embeddings (`BAAI/bge-large-en-v1.5`) and stores them in a local FAISS index (with an automatic numpy-based Cosine Similarity fallback).
   - **Sparse Search**: Implements keyword tokenization indexed via BM25 (`rank-bm25`).
   - **Fusion**: Normalizes and fuses BM25 and Vector scores.
   - **Reranker**: Employs a local cross-encoder (`BAAI/bge-reranker-base`) to select the top $N$ most relevant chunks.
3. **Agentic Loop Orchestration ([orchestrator.py](file:///home/dem/workspace/agentic_rag_4_arxiv_AG/agentic_rag/orchestrator.py))**:
   - Implements a ReAct loop enabling the LLM to decide on actions: `search_arxiv` $\rightarrow$ `download_and_index` $\rightarrow$ `retrieve_context` $\rightarrow$ `answer_user`.
   - Includes automatic error feedback loop allowing the agent to self-correct if it outputs malformed JSON.
4. **LLM Client ([llm.py](file:///home/dem/workspace/agentic_rag_4_arxiv_AG/agentic_rag/llm.py))**:
   - Integrates Google Gemini, OpenAI, Ollama (local models like `Qwen 2.5`/`Llama 3.1`), and a Mock Client.
5. **RAG Triad Evaluator ([evaluator.py](file:///home/dem/workspace/agentic_rag_4_arxiv_AG/agentic_rag/evaluator.py))**:
   - Mimics RAGAS / TruLens metrics using an LLM-as-a-Judge protocol evaluating context relevance, faithfulness (groundedness), and answer relevance.
