"""
Re-ranker module.
Supports two backends:
  1. Local cross-encoder (ms-marco-MiniLM) — free, good quality
  2. Cohere Rerank API — best quality, requires API key
"""
from __future__ import annotations

import logging

from backend.core.config import get_settings
from backend.core.exceptions import RerankerError
from backend.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """
    Local cross-encoder re-ranker.
    Uses sentence-transformers CrossEncoder for query-document scoring.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or settings.reranker_model
        logger.info("Loading cross-encoder: %s", self.model_name)
        try:
            self._model = CrossEncoder(self.model_name)
        except Exception as exc:
            raise RerankerError(f"Failed to load cross-encoder: {exc}") from exc

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Re-rank results by cross-encoder score."""
        if not results:
            return []

        k = top_k or settings.top_k_rerank
        pairs = [(query, r.text) for r in results]

        try:
            scores = self._model.predict(pairs)
        except Exception as exc:
            raise RerankerError(f"Cross-encoder prediction failed: {exc}") from exc

        for result, score in zip(results, scores):
            result.score = float(score)

        reranked = sorted(results, key=lambda r: r.score, reverse=True)[:k]
        logger.info(
            "Cross-encoder reranked %d → %d results", len(results), len(reranked)
        )
        return reranked


class CohereReranker:
    """
    Cohere Rerank API re-ranker.
    Requires COHERE_API_KEY in environment.
    """

    def __init__(self, api_key: str | None = None, model: str = "rerank-english-v3.0") -> None:
        import cohere

        self.model = model
        key = api_key or settings.cohere_api_key
        if not key:
            raise RerankerError("COHERE_API_KEY is not set")
        self._client = cohere.Client(key)
        logger.info("CohereReranker initialised with model: %s", self.model)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int | None = None
    ) -> list[SearchResult]:
        """Re-rank using Cohere Rerank API."""
        if not results:
            return []

        k = top_k or settings.top_k_rerank
        docs = [r.text for r in results]

        try:
            response = self._client.rerank(
                query=query,
                documents=docs,
                top_n=k,
                model=self.model,
            )
        except Exception as exc:
            raise RerankerError(f"Cohere rerank failed: {exc}") from exc

        reranked: list[SearchResult] = []
        for hit in response.results:
            result = results[hit.index]
            result.score = hit.relevance_score
            reranked.append(result)

        logger.info(
            "Cohere reranked %d → %d results", len(results), len(reranked)
        )
        return reranked


def get_reranker() -> CrossEncoderReranker | CohereReranker:
    """Factory: return reranker based on settings."""
    if settings.reranker_provider == "cohere":
        return CohereReranker()
    return CrossEncoderReranker()
