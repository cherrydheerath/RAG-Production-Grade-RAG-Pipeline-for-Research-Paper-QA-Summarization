"""
Ingestion API routes.
Supports ArXiv query ingestion and direct PDF upload.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.embeddings.encoder import get_encoder
from backend.ingestion.arxiv_fetcher import ArXivFetcher
from backend.ingestion.chunker import TextChunker
from backend.ingestion.document_loader import DocumentLoader
from backend.monitoring.prometheus_metrics import INGESTION_ERRORS, VECTOR_STORE_SIZE
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_store import QdrantVectorStore

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Shared singletons (initialised lazily) ────────────────────────────────────
_vector_store: QdrantVectorStore | None = None
_bm25_index: BM25Index | None = None


def get_vector_store() -> QdrantVectorStore:
    global _vector_store
    if _vector_store is None:
        encoder = get_encoder()
        _vector_store = QdrantVectorStore(embedding_dim=encoder.embedding_dim)
        _vector_store.ensure_collection()
    return _vector_store


def get_bm25() -> BM25Index:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25Index()
    return _bm25_index


# ── Request / Response schemas ────────────────────────────────────────────────
class ArXivIngestRequest(BaseModel):
    query: str = Field(..., example="attention is all you need")
    max_results: int = Field(default=10, ge=1, le=100)
    collection_name: str | None = None


class IngestResponse(BaseModel):
    status: str
    papers_fetched: int
    chunks_indexed: int
    collection: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest/arxiv", response_model=IngestResponse)
async def ingest_arxiv(request: ArXivIngestRequest) -> IngestResponse:
    """
    Fetch papers from ArXiv and index them into the vector store.
    """
    logger.info("ArXiv ingest: query=%r, max_results=%d", request.query, request.max_results)

    try:
        fetcher = ArXivFetcher(max_results=request.max_results)
        papers = fetcher.search(request.query, max_results=request.max_results)
    except Exception as exc:
        INGESTION_ERRORS.inc()
        raise HTTPException(status_code=502, detail=f"ArXiv fetch failed: {exc}")

    loader = DocumentLoader()
    chunker = TextChunker()
    encoder = get_encoder()
    vs = get_vector_store()
    bm25 = get_bm25()

    all_chunks = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        pdf_dir = Path(tmp_dir)
        pdfs = fetcher.download_batch(papers, pdf_dir, limit=request.max_results)

        metadata_map = {
            p.name: next(
                (paper for paper in papers if paper["arxiv_id"] in p.name), {}
            )
            for p in pdfs
        }

        docs = loader.load_directory(pdf_dir, metadata_map)
        all_chunks = chunker.chunk_documents(docs)

    if not all_chunks:
        raise HTTPException(status_code=422, detail="No content extracted from papers")

    texts = [c.text for c in all_chunks]
    embeddings = encoder.encode_documents(texts)

    vs.upsert_chunks(all_chunks, embeddings)
    bm25.build(all_chunks)

    VECTOR_STORE_SIZE.set(vs.count())

    return IngestResponse(
        status="success",
        papers_fetched=len(papers),
        chunks_indexed=len(all_chunks),
        collection=vs.collection_name,
    )


@router.post("/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)) -> IngestResponse:
    """
    Upload and index a single PDF file.
    """
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    logger.info("PDF upload: %s", file.filename)

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        loader = DocumentLoader()
        doc = loader.load_pdf(tmp_path, {"filename": file.filename})
        tmp_path.unlink(missing_ok=True)
    except Exception as exc:
        INGESTION_ERRORS.inc()
        raise HTTPException(status_code=422, detail=f"PDF processing failed: {exc}")

    chunker = TextChunker()
    encoder = get_encoder()
    vs = get_vector_store()
    bm25 = get_bm25()

    chunks = chunker.chunk_document(doc)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text extracted from PDF")

    texts = [c.text for c in chunks]
    embeddings = encoder.encode_documents(texts)
    vs.upsert_chunks(chunks, embeddings)
    bm25.build(chunks)

    VECTOR_STORE_SIZE.set(vs.count())

    return IngestResponse(
        status="success",
        papers_fetched=1,
        chunks_indexed=len(chunks),
        collection=vs.collection_name,
    )
