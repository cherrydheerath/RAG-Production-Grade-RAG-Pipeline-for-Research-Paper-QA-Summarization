"""
FastAPI application entrypoint for ScholarRAG.
"""
from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from backend.api.routes import health, ingest, query
from backend.core.config import get_settings
from backend.core.exceptions import ScholarRAGError
from backend.monitoring.langsmith_tracker import configure_langsmith
from backend.monitoring.prometheus_metrics import record_request

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
configure_langsmith()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ScholarRAG API",
    description="Production-Grade RAG Pipeline for Research Paper QA & Summarization",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics endpoint ───────────────────────────────────────────────
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ── Middleware: request timing + metrics ──────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):  # noqa: ANN001
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    status = str(response.status_code)
    record_request(endpoint=request.url.path, status=status, duration=duration)
    response.headers["X-Process-Time"] = f"{duration:.4f}s"
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(ScholarRAGError)
async def scholarrag_exception_handler(request: Request, exc: ScholarRAGError):  # noqa: ANN001
    logger.error("ScholarRAGError: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc), "type": type(exc).__name__})


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "ScholarRAG",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
