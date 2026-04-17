"""Central configuration for HR Assistant."""

import os

# ── LLM ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "glm-4.7"
API_BASE_URL = "https://api.z.ai/api/coding/paas/v4/"
MAX_TOKENS = 1024

# ── RAG ──────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3
MAX_DISTANCE = 2.0      # L2 distance threshold; results above this are filtered
MAX_FILE_SIZE_MB = 10
MAX_QUESTION_LENGTH = 500
MAX_HISTORY_TURNS = 5   # Recent conversation turns included in LLM context

# ── App ───────────────────────────────────────────────────────────────────────
APP_TITLE = "HR 小幫手"
APP_VERSION = "2.0"
DOCS_DIR = "docs"

# ── FAQ shortcuts shown in sidebar ────────────────────────────────────────────
FAQ_QUESTIONS = [
    "年假有幾天？",
    "病假最多可以請幾天？",
    "薪資什麼時候發放？",
    "離職需要提前多久通知？",
    "年終獎金怎麼計算？",
    "事假是否有薪？",
]

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是公司的 HR 小幫手，專門回答員工關於公司規定的問題。

規則：
- 只根據提供的文件內容回答
- 如果文件沒有提到，說「這部分文件沒有記載，建議直接聯絡 HR 部門」
- 回答要簡潔清楚，條列式呈現
- 用繁體中文回答
- 回答結尾附上參考的文件段落摘要"""
