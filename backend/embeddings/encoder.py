"""
Embedding encoder using SentenceTransformers.
Supports BGE-large-en-v1.5, E5-large, and any compatible HF model.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.core.config import get_settings
from backend.core.exceptions import EmbeddingError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingEncoder:
    """
    Wraps SentenceTransformers for batch embedding generation.

    BGE models: prepend "Represent this sentence for searching relevant passages: "
    to queries (not documents) for optimal performance.
    """

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device

        logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
        try:
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load model {self.model_name}: {exc}") from exc

        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dim: %d", self.embedding_dim)

    def encode_documents(
        self, texts: list[str], batch_size: int = 32, show_progress: bool = True
    ) -> NDArray[np.float32]:
        """Encode document texts (no prefix applied)."""
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Document encoding failed: {exc}") from exc

    def encode_query(self, query: str) -> NDArray[np.float32]:
        """Encode a single query string (BGE prefix applied if applicable)."""
        text = query
        if "bge" in self.model_name.lower():
            text = self.BGE_QUERY_PREFIX + query
        try:
            vec = self._model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            return vec[0].astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Query encoding failed: {exc}") from exc

    def encode_queries(self, queries: list[str]) -> NDArray[np.float32]:
        """Encode multiple queries."""
        if not queries:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        processed = [
            (self.BGE_QUERY_PREFIX + q if "bge" in self.model_name.lower() else q)
            for q in queries
        ]
        try:
            return self._model.encode(  # type: ignore[return-value]
                processed,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
        except Exception as exc:
            raise EmbeddingError(f"Batch query encoding failed: {exc}") from exc


@lru_cache(maxsize=1)
def get_encoder() -> EmbeddingEncoder:
    """Return cached encoder singleton."""
    return EmbeddingEncoder()
