"""Unit tests for HRKnowledgeBase."""

import pytest
from pathlib import Path

from rag_core import DocumentLoadError, HRKnowledgeBase


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


# ── ask_stream (no API call) ──────────────────────────────────────────────────

def test_ask_stream_no_docs_yields_fallback(kb):
    gen, results = kb.ask_stream("年假有幾天")
    text = "".join(gen)
    assert results == []
    assert "沒有找到" in text or "沒有" in text
