"""
Core configuration using Pydantic Settings.
All values are loaded from environment variables / .env file.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    llm_provider: Literal["openai", "anthropic", "local"] = Field(
        default="openai", alias="LLM_PROVIDER"
    )
    llm_model: str = Field(default="gpt-4o", alias="LLM_MODEL")

    # ── Embeddings ───────────────────────────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="scholarrag", alias="QDRANT_COLLECTION")

    # ── Cohere ───────────────────────────────────────────────────────────────
    cohere_api_key: str = Field(default="", alias="COHERE_API_KEY")
    reranker_provider: Literal["cross-encoder", "cohere"] = Field(
        default="cross-encoder", alias="RERANKER_PROVIDER"
    )
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL"
    )

    # ── ArXiv ────────────────────────────────────────────────────────────────
    arxiv_max_results: int = Field(default=100, alias="ARXIV_MAX_RESULTS")
    arxiv_default_query: str = Field(
        default="machine learning", alias="ARXIV_DEFAULT_QUERY"
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(default=20, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")
    dense_weight: float = Field(default=0.7, alias="DENSE_WEIGHT")
    sparse_weight: float = Field(default=0.3, alias="SPARSE_WEIGHT")

    # ── Quality Gate ─────────────────────────────────────────────────────────
    quality_min_faithfulness: float = Field(
        default=0.7, alias="QUALITY_MIN_FAITHFULNESS"
    )
    quality_min_relevancy: float = Field(default=0.6, alias="QUALITY_MIN_RELEVANCY")

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str = Field(default="", alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="scholarrag", alias="LANGCHAIN_PROJECT")

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_workers: int = Field(default=2, alias="API_WORKERS")
    api_log_level: str = Field(default="info", alias="API_LOG_LEVEL")

    # ── Frontend ─────────────────────────────────────────────────────────────
    backend_url: str = Field(default="http://localhost:8000", alias="BACKEND_URL")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
