"""Unit tests for HRKnowledgeBase."""

from pathlib import Path

import numpy as np
import pytest

from rag_core import DocumentLoadError, HRKnowledgeBase


class FakeEmbedder:
    """Small deterministic embedder so unit tests never need network/model downloads."""

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            year_leave = (
                1.0
                if "年假" in text or "七天" in text or "十天" in text
                else 0.0
            )
            sick_leave = 1.0 if "病假" in text or "三十天" in text else 0.0
            generic_policy = 1.0 if "政策" in text or "公司" in text else 0.0
            length = min(len(text) / 100.0, 1.0)
            vector = np.array(
                [year_leave, sick_leave, generic_policy, length],
                dtype="float32",
            )
            if normalize_embeddings:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            vectors.append(vector)
        return np.vstack(vectors)


@pytest.fixture(autouse=True)
def fake_embedder(monkeypatch):
    monkeypatch.setattr("rag_core.get_embedder", lambda: FakeEmbedder())


@pytest.fixture
def kb():
    return HRKnowledgeBase()


@pytest.fixture
def policy_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "policy.txt"
    doc.write_text(
        "員工到職滿一年後，每年享有七天有薪年假。\n"
        "到職滿三年後，年假增加至十天。\n"
        "病假每年最多三十天有薪假。\n",
        encoding="utf-8",
    )
    return doc


# ── add_document ──────────────────────────────────────────────────────────────

def test_add_document_returns_positive_chunk_count(kb, policy_doc):
    count = kb.add_document(str(policy_doc))
    assert count > 0


def test_add_document_updates_metadata(kb, policy_doc):
    count = kb.add_document(str(policy_doc))
    assert len(kb.metadata) == 1
    assert kb.metadata[0]["filename"] == policy_doc.name
    assert kb.metadata[0]["chunk_count"] == count
    assert kb.metadata[0]["loaded_at"] > 0


def test_add_document_tracks_sources(kb, policy_doc):
    kb.add_document(str(policy_doc))
    assert all(s == policy_doc.name for s in kb.sources)


def test_add_document_file_not_found(kb):
    with pytest.raises(DocumentLoadError, match="File not found"):
        kb.add_document("/nonexistent/missing.txt")


def test_add_document_empty_file(kb, tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n  ", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="empty"):
        kb.add_document(str(empty))


def test_add_document_rejects_unsupported_file_type(kb, tmp_path):
    unsupported = tmp_path / "policy.docx"
    unsupported.write_text("年假規定", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="Unsupported file type"):
        kb.add_document(str(unsupported))


def test_add_document_multiple_files_accumulates(kb, tmp_path):
    for i in range(3):
        f = tmp_path / f"doc{i}.txt"
        f.write_text(f"文件 {i} 的內容，包含公司政策說明。" * 5, encoding="utf-8")
        kb.add_document(str(f))
    assert len(kb.metadata) == 3
    assert len(set(kb.sources)) == 3


# ── search ────────────────────────────────────────────────────────────────────

def test_search_empty_kb_returns_empty(kb):
    assert kb.search("年假") == []


def test_search_empty_query_returns_empty(kb, policy_doc):
    kb.add_document(str(policy_doc))
    assert kb.search("   ") == []


def test_search_returns_results_after_load(kb, policy_doc):
    kb.add_document(str(policy_doc))
    results = kb.search("年假有幾天")
    assert len(results) > 0


def test_search_result_schema(kb, policy_doc):
    kb.add_document(str(policy_doc))
    results = kb.search("年假")
    for r in results:
        assert "content" in r
        assert "source" in r
        assert "distance" in r
        assert "relevance" in r


def test_search_relevance_in_range(kb, policy_doc):
    kb.add_document(str(policy_doc))
    results = kb.search("年假")
    for r in results:
        assert 0.0 <= r["relevance"] <= 1.0


def test_search_source_matches_filename(kb, policy_doc):
    kb.add_document(str(policy_doc))
    results = kb.search("病假")
    assert all(r["source"] == policy_doc.name for r in results)


def test_search_top_k_respected(kb, policy_doc):
    kb.add_document(str(policy_doc))
    results = kb.search("年假", top_k=1)
    assert len(results) <= 1


def test_search_non_positive_top_k_returns_empty(kb, policy_doc):
    kb.add_document(str(policy_doc))
    assert kb.search("年假", top_k=0) == []
