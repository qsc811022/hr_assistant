"""RAG engine: document ingestion, vector indexing, and semantic search."""

import logging
import time
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    MAX_FILE_SIZE_MB,
    TOP_K,
)

load_dotenv()
logger = logging.getLogger(__name__)

_embedder: SentenceTransformer | None = None
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}


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
        self.metadata: list[dict] = []
        self.index: faiss.IndexFlatL2 | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def add_document(self, path: str) -> int:
        """Load a document into the knowledge base. Returns chunk count."""
        file_path = Path(path)

        if not file_path.exists():
            raise DocumentLoadError(f"File not found: {path}")

        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise DocumentLoadError(
                f"Unsupported file type: {file_path.suffix} (allowed: {allowed})"
            )

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
            raise DocumentLoadError(f"Failed to read {file_path.name}: {exc}") from exc

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
        query = query.strip()
        if self.index is None or not self.chunks or not query or top_k <= 0:
            return []

        embedder = get_embedder()
        query_vec = embedder.encode([query], normalize_embeddings=True)
        limit = min(top_k, len(self.chunks))
        distances, indices = self.index.search(
            np.array(query_vec, dtype="float32"), limit
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise DocumentLoadError("pypdf not installed; cannot read PDF files") from exc
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        return path.read_text(encoding="utf-8")

    def _split_text(self, text: str) -> list[str]:
        chunks, start = [], 0
        step = CHUNK_SIZE - CHUNK_OVERLAP
        if step <= 0:
            raise DocumentLoadError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE]
            if chunk.strip():
                chunks.append(chunk)
            start += step
        return chunks

    def _rebuild_index(self) -> None:
        if not self.chunks:
            return
        embedder = get_embedder()
        embeddings = embedder.encode(
            self.chunks, normalize_embeddings=True, show_progress_bar=False
        )
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))
