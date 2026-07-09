"""
Query API route — the main RAG pipeline endpoint.
Orchestrates: embed → hybrid retrieve → rerank → generate → quality check.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.embeddings.encoder import get_encoder
from backend.generation.llm_client import LLMClient
from backend.generation.prompt_builder import PromptBuilder
from backend.generation.quality_checker import QualityChecker
from backend.monitoring.langsmith_tracker import trace_pipeline_step
from backend.monitoring.prometheus_metrics import (
    QUALITY_GATE_RESULTS,
    RERANKED_CHUNKS,
    RETRIEVAL_CANDIDATES,
    record_token_usage,
)
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.reranker import get_reranker
from backend.retrieval.vector_store import QdrantVectorStore

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, example="What is attention mechanism in transformers?")
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = Field(default=True)
    collection_name: str | None = None


class SourceDoc(BaseModel):
    index: int
    source: str
    score: float
    snippet: str
    metadata: dict


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]
    quality_score: float
    quality_passed: bool
    latency_seconds: float
    model: str
    tokens_used: int


# ── Pipeline state (singletons) ───────────────────────────────────────────────
_pipeline_cache: dict = {}


def _get_pipeline():
    if "vs" not in _pipeline_cache:
        from backend.api.routes.ingest import get_bm25, get_vector_store
        _pipeline_cache["vs"] = get_vector_store()
        _pipeline_cache["bm25"] = get_bm25()
        _pipeline_cache["encoder"] = get_encoder()
        _pipeline_cache["reranker"] = get_reranker()
        _pipeline_cache["retriever"] = HybridRetriever(
            vector_store=_pipeline_cache["vs"],
            bm25_index=_pipeline_cache["bm25"],
        )
        _pipeline_cache["prompt_builder"] = PromptBuilder()
        _pipeline_cache["llm"] = LLMClient()
        _pipeline_cache["quality_checker"] = QualityChecker()
    return _pipeline_cache


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Run the full RAG pipeline:
    embed → hybrid retrieve → rerank → LLM generate → quality check
    """
    t_start = time.perf_counter()
    logger.info("Query: %r", request.question)

    try:
        pipe = _get_pipeline()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Pipeline not ready: {exc}")

    # 1. Embed query
    with trace_pipeline_step("embed_query", {"question": request.question}):
        query_vector = pipe["encoder"].encode_query(request.question)

    # 2. Hybrid retrieval
    with trace_pipeline_step("hybrid_retrieve"):
        candidates = pipe["retriever"].retrieve(
            query=request.question,
            query_vector=query_vector,
            top_k=request.top_k * 4,
        )
        RETRIEVAL_CANDIDATES.observe(len(candidates))
        logger.info("Retrieved %d candidates", len(candidates))

    # 3. Re-ranking
    if request.rerank and candidates:
        with trace_pipeline_step("rerank"):
            candidates = pipe["reranker"].rerank(
                query=request.question,
                results=candidates,
                top_k=request.top_k,
            )
    else:
        candidates = candidates[: request.top_k]

    RERANKED_CHUNKS.observe(len(candidates))

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Please ingest papers first.",
        )

    # 4. Prompt construction
    with trace_pipeline_step("build_prompt"):
        system_prompt, user_prompt = pipe["prompt_builder"].build(
            question=request.question,
            context_chunks=candidates,
        )
        sources_meta = pipe["prompt_builder"].build_sources_list(candidates)

    # 5. LLM generation
    with trace_pipeline_step("llm_generate"):
        llm_response = pipe["llm"].generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        record_token_usage(llm_response.prompt_tokens, llm_response.completion_tokens)

    # 6. Quality gate
    quality = pipe["quality_checker"].check(
        answer=llm_response.answer, question=request.question
    )
    gate_label = "pass" if quality.passed else "fail"
    QUALITY_GATE_RESULTS.labels(result=gate_label).inc()

    latency = time.perf_counter() - t_start

    return QueryResponse(
        question=request.question,
        answer=llm_response.answer,
        sources=[SourceDoc(**s) for s in sources_meta],
        quality_score=quality.score,
        quality_passed=quality.passed,
        latency_seconds=round(latency, 3),
        model=llm_response.model,
        tokens_used=llm_response.total_tokens,
    )
