"""
Qdrant vector store integration.
Handles collection creation, upsert, and dense vector search.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from backend.core.config import get_settings
from backend.core.exceptions import VectorStoreError
from backend.ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class SearchResult:
    """Single result from vector search."""

    chunk_id: str
    text: str
    source: str
    metadata: dict
    score: float


class QdrantVectorStore:
    """
    Self-hosted Qdrant vector store.
    Manages collections and provides dense similarity search.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
        embedding_dim: int = 1024,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.api_key = api_key or settings.qdrant_api_key or None
        self.collection_name = collection_name or settings.qdrant_collection
        self.embedding_dim = embedding_dim

        try:
            if not self.url or not self.url.startswith("http"):
                logger.info("Initializing local disk-based Qdrant client in 'data/qdrant'")
                self._client = QdrantClient(path="data/qdrant")
            else:
                self._client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=30,
                )
                logger.info("Connected to Qdrant at %s", self.url)
        except Exception as exc:
            raise VectorStoreError(f"Cannot connect to Qdrant: {exc}") from exc

    def ensure_collection(self) -> None:
        """Create collection if it does not exist."""
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name in existing:
            logger.info("Collection '%s' already exists", self.collection_name)
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.embedding_dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info(
            "Created Qdrant collection '%s' (dim=%d)", self.collection_name, self.embedding_dim
        )

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ) -> int:
        """
        Upsert chunks with their embeddings into Qdrant.

        Args:
            chunks: List of DocumentChunk objects
            embeddings: Numpy array of shape (len(chunks), embedding_dim)

        Returns:
            Number of upserted points
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        points = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id))
            payload = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                **chunk.metadata,
            }
            points.append(
                qdrant_models.PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        BATCH = 100
        total = 0
        for i in range(0, len(points), BATCH):
            batch = points[i : i + BATCH]
            self._client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )
            total += len(batch)
            logger.debug("Upserted batch %d/%d", i + BATCH, len(points))

        logger.info("Upserted %d points into '%s'", total, self.collection_name)
        return total

    def dense_search(
        self,
        query_vector: np.ndarray,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Perform dense cosine similarity search."""
        k = top_k or settings.top_k_retrieval
        try:
            hits = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Dense search failed: {exc}") from exc

        return [
            SearchResult(
                chunk_id=hit.payload.get("chunk_id", ""),
                text=hit.payload.get("text", ""),
                source=hit.payload.get("source", ""),
                metadata={k: v for k, v in hit.payload.items() if k not in {"text", "chunk_id", "source"}},
                score=hit.score,
            )
            for hit in hits
        ]

    def delete_collection(self) -> None:
        """Drop the collection (use with caution)."""
        self._client.delete_collection(self.collection_name)
        logger.warning("Deleted collection '%s'", self.collection_name)

    def count(self) -> int:
        """Return number of vectors in the collection."""
        info = self._client.get_collection(self.collection_name)
        return info.points_count or 0
