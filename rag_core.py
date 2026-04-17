"""RAG engine: document ingestion, embedding, search, and streaming LLM responses."""

import logging
import os
import time

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pathlib import Path
from sentence_transformers import SentenceTransformer

from config import (
    API_BASE_URL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    MAX_FILE_SIZE_MB,
    MAX_HISTORY_TURNS,
    MAX_TOKENS,
    MODEL_NAME,
    SYSTEM_PROMPT,
    TOP_K,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Module-level singletons — loaded once, reused across sessions
_client: OpenAI | None = None
_embedder: SentenceTransformer | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("ZAI_API_KEY")
        if not api_key:
            raise EnvironmentError("ZAI_API_KEY not set in environment variables")
        _client = OpenAI(api_key=api_key, base_url=API_BASE_URL)
    return _client


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


class DocumentLoadError(Exception):
    """Raised when a document cannot be loaded or parsed."""


class HRKnowledgeBase:
    def __init__(self) -> None:
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self.metadata: list[dict] = []   # [{filename, loaded_at, chunk_count}]
        self.index: faiss.IndexFlatL2 | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(self, path: str) -> int:
        """Load a document into the knowledge base. Returns number of chunks added."""
        file_path = Path(path)

        if not file_path.exists():
            raise DocumentLoadError(f"File not found: {path}")

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise DocumentLoadError(
                f"File too large: {size_mb:.1f} MB (max {MAX_FILE_SIZE_MB} MB)"
            )

        try:
            text = self._read_file(file_path)
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(
                f"Failed to read {file_path.name}: {exc}"
            ) from exc

        if not text.strip():
            raise DocumentLoadError(f"File is empty: {file_path.name}")

        new_chunks = self._split_text(text)
        chunk_count = len(new_chunks)

        self.chunks.extend(new_chunks)
        self.sources.extend([file_path.name] * chunk_count)
        self.metadata.append(
            {
                "filename": file_path.name,
                "loaded_at": time.time(),
                "chunk_count": chunk_count,
            }
        )

        self._rebuild_index()
        logger.info("Loaded %s: %d chunks", file_path.name, chunk_count)
        return chunk_count

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Return top-k semantically relevant chunks with source and relevance score."""
        if self.index is None or not self.chunks:
            return []

        embedder = get_embedder()
        query_vec = embedder.encode([query])
        distances, indices = self.index.search(
            np.array(query_vec, dtype="float32"), top_k
        )

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks) and float(dist) <= MAX_DISTANCE:
                results.append(
                    {
                        "content": self.chunks[idx],
                        "source": self.sources[idx],
                        "distance": float(dist),
                        "relevance": max(0.0, 1.0 - float(dist) / MAX_DISTANCE),
                    }
                )
        return results

    def ask_stream(
        self,
        question: str,
        history: list[dict] | None = None,
    ) -> tuple:
        """Stream an LLM answer grounded in retrieved chunks.

        Returns (token_generator, search_results). Raises on API failure.
        """
        results = self.search(question)
        if not results:
            def _no_match():
                yield "目前沒有找到相關文件內容，建議直接聯絡 HR 部門。"

            return _no_match(), []

        context = "\n\n".join(
            f"【來源：{r['source']}】\n{r['content']}" for r in results
        )

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if history:
            for turn in history[-(MAX_HISTORY_TURNS * 2):]:
                if turn["role"] in ("user", "assistant"):
                    messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append(
            {
                "role": "user",
                "content": f"參考資料：\n{context}\n\n問題：{question}",
            }
        )

        stream = self._call_api_with_retry(messages)

        def _generator():
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return _generator(), results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise DocumentLoadError("pypdf not installed; cannot read PDF files") from exc
            reader = PdfReader(str(path))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        return path.read_text(encoding="utf-8")

    def _split_text(self, text: str) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE]
            if chunk.strip():
                chunks.append(chunk)
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks

    def _rebuild_index(self) -> None:
        if not self.chunks:
            return
        embedder = get_embedder()
        embeddings = embedder.encode(self.chunks, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

    def _call_api_with_retry(self, messages: list[dict], retries: int = 3):
        client = get_client()
        last_error: Exception | None = None

        for attempt in range(retries):
            try:
                return client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    stream=True,
                )
            except RateLimitError as exc:
                wait = 2**attempt
                logger.warning(
                    "Rate limited; retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries
                )
                time.sleep(wait)
                last_error = exc
            except (APIConnectionError, APIError) as exc:
                logger.error("API error on attempt %d: %s", attempt + 1, exc)
                last_error = exc
                break

        raise last_error or RuntimeError("LLM API call failed after retries")
