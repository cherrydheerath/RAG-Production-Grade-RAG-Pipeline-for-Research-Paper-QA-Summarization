# ScholarRAG — Architecture Documentation

## System Overview

ScholarRAG is a production-grade RAG (Retrieval-Augmented Generation) pipeline for research paper QA and summarization. The system is designed with modularity, observability, and evaluation at its core.

## Pipeline Stages

### Stage 1: Data Ingestion
**Module:** `backend/ingestion/`

- **ArXiv Fetcher** (`arxiv_fetcher.py`): Queries the ArXiv API using the `arxiv` Python SDK. Fetches paper metadata and downloads PDFs. Rate-limited to comply with ArXiv's terms.
- **Document Loader** (`document_loader.py`): Uses `unstructured.io` for high-quality PDF parsing, handling multi-column layouts, tables, and headers. Falls back to `pypdf` for simple documents.

### Stage 2: Text Chunking
**Module:** `backend/ingestion/chunker.py`

Uses LangChain's `RecursiveCharacterTextSplitter` with `["\n\n", "\n", ". ", " ", ""]` separators to preserve semantic boundaries. Default: 512 tokens, 64 overlap.

### Stage 3: Embedding Generation
**Module:** `backend/embeddings/encoder.py`

Uses `SentenceTransformers` with BGE-large-en-v1.5 (1024-dim) or E5-large. BGE models require a query prefix for asymmetric retrieval: `"Represent this sentence for searching relevant passages: "`.

### Stage 4: Vector Store (Qdrant)
**Module:** `backend/retrieval/vector_store.py`

Self-hosted Qdrant instance via Docker. Uses cosine similarity with HNSW indexing. Points are stored with full metadata payloads for citation retrieval.

### Stage 5: BM25 Keyword Index
**Module:** `backend/retrieval/bm25_index.py`

In-memory BM25Okapi index from `rank-bm25`. Provides sparse retrieval for exact keyword matching. Can be persisted to disk with pickle serialization.

### Stage 6: Hybrid Retrieval
**Module:** `backend/retrieval/hybrid_retriever.py`

Implements **Reciprocal Rank Fusion (RRF)**:

```
score(d) = Σ weight_i / (k + rank_i)
```

where `k=60` (Cormack et al. 2009), `dense_weight=0.7`, `sparse_weight=0.3`.

### Stage 7: Re-ranking
**Module:** `backend/retrieval/reranker.py`

Two options:
- **CrossEncoderReranker**: Local `ms-marco-MiniLM-L-6-v2` cross-encoder. Free, runs on CPU.
- **CohereReranker**: `rerank-english-v3.0` API. Higher quality, requires API key.

### Stage 8: LLM Generation
**Module:** `backend/generation/`

- **PromptBuilder**: Constructs citation-aware prompts with numbered source excerpts. Instructs the LLM to cite every claim.
- **LLMClient**: Unified client for OpenAI (GPT-4o) and Anthropic (Claude). Retries on transient failures with exponential backoff.

### Stage 9: Quality Gate
**Module:** `backend/generation/quality_checker.py`

Rule-based gate checking: non-empty, minimum length (50 chars), has citations `[Source N]`, not a refusal response. Score = fraction of checks passed.

### Stage 10: Evaluation (RAGAS)
**Module:** `backend/evaluation/ragas_evaluator.py`

RAGAS metrics:
- **Faithfulness**: Are claims in the answer supported by the context?
- **Answer Relevancy**: Is the answer relevant to the question?
- **Context Precision**: What fraction of retrieved context is relevant?
- **Context Recall**: What fraction of ground truth is covered?

### Stage 11: API (FastAPI)
**Module:** `backend/api/`

RESTful API with:
- `POST /api/v1/ingest/arxiv` — ArXiv ingestion
- `POST /api/v1/ingest/pdf` — PDF upload
- `POST /api/v1/query` — RAG query
- `GET /api/v1/health` — Health check
- `GET /metrics` — Prometheus metrics

### Stage 12: Frontend (Streamlit)
**Module:** `frontend/streamlit_app.py`

Three-tab UI: Query (ask questions), Ingest (ArXiv/PDF), Evaluate (RAGAS results viewer).

### Stage 13: Monitoring
- **LangSmith**: Traces LLM calls, retrieval steps, and token usage.
- **Prometheus**: Metrics for request rate, latency, token usage, quality gate results.
- **Grafana**: Pre-configured dashboard connecting to Prometheus.

### Stage 14: CI/CD
- **GitHub Actions CI**: Lint (Ruff + Black) → Unit tests (pytest + coverage) → Docker build
- **GitHub Actions CD**: Build and push images to GHCR on main branch / tags

## Data Flow Diagram

```
User Query
    │
    ▼
[Embed Query] ─── BGE-large encoder ─────────────────────────┐
                                                              │
[BM25 Search] ◄─── rank-bm25 index ◄─── ingested chunks     │
    │                                                         │
    ├──────────────────────────────────────────────────────   │
    │                                                         ▼
    └──────────────► [Qdrant Dense Search] ◄─── query vector─┘
                              │
                              ▼
                    [RRF Fusion Merge]
                              │
                              ▼
                    [Cross-Encoder Rerank]
                              │
                              ▼
                    [Prompt Builder]
                    (context + citations)
                              │
                              ▼
                    [LLM Generation]
                    (GPT-4 / Claude)
                              │
                              ▼
                    [Quality Gate]
                              │
                              ▼
                    [Answer + Sources]
                    (returned to user)
```
