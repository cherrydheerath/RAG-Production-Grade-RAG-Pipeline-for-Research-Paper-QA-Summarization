"""Custom exceptions for ScholarRAG."""
from __future__ import annotations


class ScholarRAGError(Exception):
    """Base exception."""


class IngestionError(ScholarRAGError):
    """Raised when document ingestion fails."""


class EmbeddingError(ScholarRAGError):
    """Raised when embedding generation fails."""


class RetrievalError(ScholarRAGError):
    """Raised when retrieval fails."""


class GenerationError(ScholarRAGError):
    """Raised when LLM generation fails."""


class QualityCheckError(ScholarRAGError):
    """Raised when the generated answer fails the quality gate."""


class RerankerError(ScholarRAGError):
    """Raised when reranking fails."""


class VectorStoreError(ScholarRAGError):
    """Raised when vector store operations fail."""
