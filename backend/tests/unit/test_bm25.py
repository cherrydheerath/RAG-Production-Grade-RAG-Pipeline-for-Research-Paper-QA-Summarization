"""Unit tests for BM25Index."""
from __future__ import annotations

import pytest

from backend.ingestion.chunker import DocumentChunk
from backend.retrieval.bm25_index import BM25Index


def make_chunks(texts: list[str]) -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"chunk_{i}",
            text=t,
            source="test.pdf",
            metadata={},
            chunk_index=i,
            total_chunks=len(texts),
        )
        for i, t in enumerate(texts)
    ]


def test_build_and_search():
    bm25 = BM25Index()
    chunks = make_chunks([
        "attention mechanism in transformers",
        "convolutional neural network image classification",
        "BERT language model pretraining",
    ])
    bm25.build(chunks)

    results = bm25.search("attention transformers", top_k=2)
    assert len(results) > 0
    assert results[0].text == "attention mechanism in transformers"


def test_search_empty_index():
    bm25 = BM25Index()
    results = bm25.search("anything", top_k=5)
    assert results == []


def test_build_empty():
    bm25 = BM25Index()
    bm25.build([])
    assert bm25.search("query") == []


def test_top_k_respected():
    bm25 = BM25Index()
    chunks = make_chunks([f"document about topic {i}" for i in range(20)])
    bm25.build(chunks)
    results = bm25.search("document topic", top_k=5)
    assert len(results) <= 5


def test_scores_descending():
    bm25 = BM25Index()
    chunks = make_chunks([
        "neural network deep learning",
        "random unrelated text here",
        "deep learning neural architecture",
    ])
    bm25.build(chunks)
    results = bm25.search("deep learning neural", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
