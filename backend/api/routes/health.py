"""Health check route."""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend.core.config import get_settings

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    status: str
    version: str
    qdrant_url: str
    llm_provider: str
    embedding_model: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns service health and configuration summary."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        qdrant_url=settings.qdrant_url,
        llm_provider=settings.llm_provider,
        embedding_model=settings.embedding_model,
    )
