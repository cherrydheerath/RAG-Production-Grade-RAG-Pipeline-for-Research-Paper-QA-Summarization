"""
Prometheus metrics for ScholarRAG API.
Exposes request count, latency, retrieval sizes, and token usage.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, Summary

# ── Request metrics ───────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "scholarrag_requests_total",
    "Total number of API requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "scholarrag_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# ── Pipeline metrics ──────────────────────────────────────────────────────────
RETRIEVAL_CANDIDATES = Histogram(
    "scholarrag_retrieval_candidates",
    "Number of candidates before reranking",
    buckets=[5, 10, 15, 20, 30, 50],
)

RERANKED_CHUNKS = Histogram(
    "scholarrag_reranked_chunks",
    "Number of chunks after reranking",
    buckets=[1, 2, 3, 5, 10],
)

LLM_TOKENS_USED = Counter(
    "scholarrag_llm_tokens_total",
    "Total LLM tokens consumed",
    ["type"],  # prompt | completion
)

QUALITY_GATE_RESULTS = Counter(
    "scholarrag_quality_gate_total",
    "Quality gate pass/fail count",
    ["result"],  # pass | fail
)

# ── System metrics ────────────────────────────────────────────────────────────
VECTOR_STORE_SIZE = Gauge(
    "scholarrag_vector_store_documents",
    "Number of documents in the vector store",
)

INGESTION_ERRORS = Counter(
    "scholarrag_ingestion_errors_total",
    "Total ingestion errors",
)


def record_request(endpoint: str, status: str, duration: float) -> None:
    """Helper to record a completed request."""
    REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


def record_token_usage(prompt_tokens: int, completion_tokens: int) -> None:
    """Record LLM token usage."""
    LLM_TOKENS_USED.labels(type="prompt").inc(prompt_tokens)
    LLM_TOKENS_USED.labels(type="completion").inc(completion_tokens)
