"""
Hybrid retriever combining dense (Qdrant) + sparse (BM25) search
using Reciprocal Rank Fusion (RRF) for score merging.
"""
from __future__ import annotations

import logging

import numpy as np

from backend.core.config import get_settings
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_store import QdrantVectorStore, SearchResult

logger = logging.getLogger(__name__)
settings = get_settings()

RRF_K = 60  # RRF constant (Cormack et al. 2009)


class HybridRetriever:
    """
    Fuses dense and sparse retrieval results via Reciprocal Rank Fusion.

    RRF score = Σ 1 / (k + rank_i)

    Dense results are fetched from Qdrant; sparse from BM25.
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        bm25_index: BM25Index,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        top_k: int | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.dense_weight = dense_weight if dense_weight is not None else settings.dense_weight
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.sparse_weight
        self.top_k = top_k or settings.top_k_retrieval

    def retrieve(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Perform hybrid retrieval.

        Args:
            query: Raw query string (for BM25)
            query_vector: Embedded query vector (for Qdrant)
            top_k: Number of final results

        Returns:
            Ranked list of SearchResults
        """
        k = top_k or self.top_k

        # --- Dense results ------------------------------------------------
        dense_results = self.vector_store.dense_search(query_vector, top_k=k * 2)
        logger.debug("Dense results: %d", len(dense_results))

        # --- Sparse results -----------------------------------------------
        sparse_results = self.bm25_index.search(query, top_k=k * 2)
        logger.debug("Sparse results: %d", len(sparse_results))

        # --- Reciprocal Rank Fusion ----------------------------------------
        fused = self._rrf_merge(dense_results, sparse_results)

        # Sort by fused score descending, take top_k
        fused_sorted = sorted(fused.values(), key=lambda r: r.score, reverse=True)[:k]
        logger.info("Hybrid retrieval: %d final candidates", len(fused_sorted))
        return fused_sorted

    def _rrf_merge(
        self,
        dense: list[SearchResult],
        sparse: list[SearchResult],
    ) -> dict[str, SearchResult]:
        """Apply RRF and return merged dict keyed by chunk_id."""
        scores: dict[str, float] = {}
        registry: dict[str, SearchResult] = {}

        def apply_rrf(results: list[SearchResult], weight: float) -> None:
            for rank, result in enumerate(results, start=1):
                cid = result.chunk_id
                rrf_score = weight * (1.0 / (RRF_K + rank))
                scores[cid] = scores.get(cid, 0.0) + rrf_score
                if cid not in registry:
                    registry[cid] = result

        apply_rrf(dense, self.dense_weight)
        apply_rrf(sparse, self.sparse_weight)

        # Assign merged scores
        for cid, result in registry.items():
            result.score = scores[cid]

        return registry
