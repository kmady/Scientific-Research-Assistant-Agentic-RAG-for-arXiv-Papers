import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
for directory in [DATA_DIR, PDF_DIR, CACHE_DIR, INDEX_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # gemini, openai, ollama
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")  # qwen2.5:7b, llama3.1, mistral

# Embedding Configuration
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")  # local, gemini, openai
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))  # 1024 for bge-large-en-v1.5, 1536 for OpenAI

# Reranking Configuration
USE_RERANKER = os.getenv("USE_RERANKER", "true").lower() == "true"
LOCAL_RERANKER_MODEL = os.getenv("LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-base")

# Retrieval & Hybrid Search Configuration
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.7"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "30"))  # Retrieve top K candidates for reranking
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "8"))       # Pass top N to the LLM

# Chunking Parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
