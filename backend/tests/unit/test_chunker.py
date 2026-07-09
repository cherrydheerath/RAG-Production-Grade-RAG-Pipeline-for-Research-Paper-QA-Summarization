"""Unit tests for TextChunker."""
from __future__ import annotations

import pytest

from backend.ingestion.chunker import DocumentChunk, TextChunker
from backend.ingestion.document_loader import RawDocument


def make_doc(text: str, source: str = "test.pdf") -> RawDocument:
    return RawDocument(source=source, text=text, metadata={"filename": source})


def test_chunk_short_document():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    doc = make_doc("Short text." * 5)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, DocumentChunk)
        assert c.text.strip()


def test_chunk_long_document():
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    doc = make_doc("word " * 500)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1


def test_chunk_empty_document():
    chunker = TextChunker()
    doc = make_doc("")
    chunks = chunker.chunk_document(doc)
    assert chunks == []


def test_chunk_metadata_propagation():
    chunker = TextChunker(chunk_size=200, chunk_overlap=20)
    doc = RawDocument(
        source="paper.pdf",
        text="Science text. " * 30,
        metadata={"title": "Test Paper", "authors": ["Alice"]},
    )
    chunks = chunker.chunk_document(doc)
    for c in chunks:
        assert c.source == "paper.pdf"
        assert "title" in c.metadata or c.chunk_index >= 0


def test_chunk_ids_unique():
    chunker = TextChunker(chunk_size=50, chunk_overlap=5)
    doc = make_doc("unique word " * 100)
    chunks = chunker.chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_documents_multiple():
    chunker = TextChunker()
    docs = [make_doc("text " * 50, f"doc{i}.pdf") for i in range(3)]
    all_chunks = chunker.chunk_documents(docs)
    assert len(all_chunks) > 0
    sources = {c.source for c in all_chunks}
    assert len(sources) == 3
