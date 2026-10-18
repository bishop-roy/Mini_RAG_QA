"""Unit tests for the core logic. No Ollama or model download needed."""

import numpy as np
import pytest

import rag


# ---------- Chunking ----------
def test_chunk_text_respects_size_and_overlaps():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = rag.chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # The start of each chunk should appear in the end of the previous one.
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[:5] in prev


def test_chunk_text_short_and_empty():
    assert rag.chunk_text("hello world", 100, 10) == ["hello world"]
    assert rag.chunk_text("   \n  ", 100, 10) == []


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        rag.chunk_text("abc", chunk_size=10, overlap=10)


# ---------- Metadata ----------
def test_build_chunks_keeps_source_and_page():
    pages = [
        {"page": 1, "text": "First page text."},
        {"page": 2, "text": "Second page text."},
    ]
    chunks = rag.build_chunks("report.pdf", pages)
    assert [c["page"] for c in chunks] == [1, 2]
    assert all(c["source"] == "report.pdf" for c in chunks)


def test_extract_pages_txt_has_no_page_number():
    pages = rag.extract_pages("notes.txt", b"Some text here.")
    assert pages == [{"page": None, "text": "Some text here."}]


def test_extract_pages_errors():
    with pytest.raises(rag.RagError):
        rag.extract_pages("empty.txt", b"   \n ")
    with pytest.raises(rag.RagError):
        rag.extract_pages("image.png", b"data")


# ---------- Cosine similarity ----------
def test_cosine_similarity_known_values():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, 0.0]])
    scores = rag.cosine_similarity(np.array([1.0, 0.0]), matrix)
    assert np.allclose(scores, [1.0, 0.0, -1.0, 0.0])  # zero vector scores 0, not NaN


# ---------- Retrieval and generation (model calls mocked) ----------
def test_retrieve_returns_best_chunk_first(monkeypatch):
    chunks = [
        {"text": "cats", "source": "a.txt", "page": None},
        {"text": "dogs", "source": "b.txt", "page": 3},
    ]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(rag, "embed_texts", lambda texts, model=None: np.array([[0.0, 1.0]]))
    hits = rag.retrieve("question?", chunks, embeddings, k=2)
    assert hits[0][0]["source"] == "b.txt"
    assert hits[0][1] > hits[1][1]


def test_build_messages_contains_sources_question_and_rules():
    hits = [({"text": "Books last 21 days.", "source": "guide.pdf", "page": 2}, 0.9)]
    messages = rag.build_messages("How long?", hits)
    assert messages[0]["role"] == "system"
    assert "untrusted" in messages[0]["content"]
    user = messages[1]["content"]
    assert "guide.pdf, page 2" in user
    assert "Books last 21 days." in user
    assert "Question: How long?" in user


def test_answer_question_uses_mocked_model(monkeypatch):
    monkeypatch.setattr(
        rag.ollama, "chat", lambda **kwargs: {"message": {"content": " 21 days [1] "}}
    )
    hits = [({"text": "Books last 21 days.", "source": "g.txt", "page": None}, 0.9)]
    assert rag.answer_question("How long?", hits) == "21 days [1]"


def test_connection_error_becomes_friendly_message(monkeypatch):
    def fail(**kwargs):
        raise ConnectionError("refused")

    monkeypatch.setattr(rag.ollama, "chat", fail)
    hits = [({"text": "x", "source": "g.txt", "page": None}, 0.9)]
    with pytest.raises(rag.RagError, match="Cannot reach Ollama"):
        rag.answer_question("q?", hits)
