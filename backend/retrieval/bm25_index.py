"""
BM25 keyword index for sparse retrieval.
Uses rank-bm25 library. Index is kept in memory and can be
serialised to disk for persistence.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from backend.ingestion.chunker import DocumentChunk
from backend.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> list[str]:
    """Simple whitespace + lowercase tokeniser."""
    return text.lower().split()


class BM25Index:
    """
    In-memory BM25 index backed by rank-bm25.

    Stores all chunks for result materialisation.
    """

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index from a list of chunks."""
        if not chunks:
            logger.warning("BM25: no chunks to index")
            return

        self._chunks = chunks
        corpus = [_tokenise(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus)
        logger.info("BM25 index built: %d documents", len(chunks))

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """Retrieve top-k chunks by BM25 score."""
        if self._bm25 is None:
            logger.warning("BM25 index is empty — call build() first")
            return []

        tokens = _tokenise(query)
        scores = self._bm25.get_scores(tokens)

        # Pair (score, index), sort descending
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    metadata=chunk.metadata,
                    score=float(score),
                )
            )

        logger.debug("BM25 search returned %d results for query: %r", len(results), query)
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Pickle the index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path) -> None:
        """Load a pickled index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunks = data["chunks"]
        self._bm25 = data["bm25"]
        logger.info("BM25 index loaded from %s (%d docs)", path, len(self._chunks))
