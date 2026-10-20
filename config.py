"""Settings. Everything stays on this machine (Ollama on 127.0.0.1, SQLite file)."""
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"      # private copies of ingested files
DB_PATH = DATA_DIR / "index.sqlite"
MODELS_DIR = APP_DIR / "models"        # bundled GGUF model files for standalone zero-prerequisite mode
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Standalone in-process GGUF models (used when present in models/ without requiring Ollama)
LOCAL_CHAT_GGUF = MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
LOCAL_EMBED_GGUF = MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf"

# Fallback Ollama service settings
OLLAMA_HOST = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:3b"              # for clearly better detail: "qwen2.5:7b" (~4.7 GB, 8 GB RAM)
EMBED_MODEL = "nomic-embed-text"       # use bge-m3 for multilingual documents

# Chunking: one chunk = one section of a page (heading + its paragraphs)
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 150                    # only used when a single paragraph is longer than CHUNK_CHARS

# Retrieval
TOP_K_DENSE = 30
TOP_K_BM25 = 30
TOP_K_CANDIDATES = 20                  # passages found by search
TOP_K_FINAL = 8                        # passages given to the model after re-ranking
RRF_K = 60
MIN_DENSE_SCORE = 0.28                 # below this the question is treated as "not in documents"

# Accuracy helpers (each costs one extra short model call; turn off if too slow)
USE_QUERY_EXPANSION = True             # rewrites the question into alternative search queries
USE_RERANK = True                      # the model picks the passages that really answer the question
USE_FACT_CHECK = True                  # the model flags statements not supported by the sources

TEMPERATURE = 0.1
NUM_CTX = 6144
HISTORY_TURNS = 4