"""Central configuration for HR Assistant."""

# ── RAG ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
MAX_DISTANCE = 2.0      # FAISS L2 threshold for unit-norm vectors; 2.0 = cos_sim >= 0
MAX_FILE_SIZE_MB = 10
MAX_QUESTION_LENGTH = 500
MAX_HISTORY_TURNS = 5

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE = "文件問答助手"
APP_VERSION = "3.0"
DOCS_DIR = "docs"
