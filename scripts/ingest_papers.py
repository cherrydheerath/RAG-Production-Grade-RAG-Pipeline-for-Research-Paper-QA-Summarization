#!/usr/bin/env python
"""
CLI script to ingest ArXiv papers into ScholarRAG.

Usage:
    python scripts/ingest_papers.py --query "transformers attention" --max 20
    python scripts/ingest_papers.py --pdf ./data/papers/paper.pdf
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.embeddings.encoder import EmbeddingEncoder
from backend.ingestion.arxiv_fetcher import ArXivFetcher
from backend.ingestion.chunker import TextChunker
from backend.ingestion.document_loader import DocumentLoader
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.vector_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def ingest_arxiv(query: str, max_results: int, output_dir: Path) -> None:
    logger.info("=== ArXiv Ingestion: %r (max=%d) ===", query, max_results)

    fetcher = ArXivFetcher(max_results=max_results)
    papers = fetcher.search(query, max_results=max_results)
    logger.info("Found %d papers", len(papers))

    pdfs = fetcher.download_batch(papers, output_dir)
    logger.info("Downloaded %d PDFs to %s", len(pdfs), output_dir)

    _index_pdfs(pdfs, papers)


def ingest_pdf(pdf_path: Path) -> None:
    logger.info("=== PDF Ingestion: %s ===", pdf_path)
    _index_pdfs([pdf_path], [])


def _index_pdfs(pdf_paths: list[Path], papers_meta: list[dict]) -> None:
    loader = DocumentLoader()
    chunker = TextChunker()
    encoder = EmbeddingEncoder()
    vs = QdrantVectorStore(embedding_dim=encoder.embedding_dim)
    bm25 = BM25Index()

    vs.ensure_collection()

    meta_map = {p["arxiv_id"]: p for p in papers_meta}

    all_chunks = []
    for pdf in pdf_paths:
        arxiv_id = pdf.stem.replace("_", "/")
        meta = meta_map.get(arxiv_id, {"filename": pdf.name})
        try:
            doc = loader.load_pdf(pdf, meta)
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf.name, exc)

    if not all_chunks:
        logger.warning("No chunks produced. Check your PDFs.")
        return

    logger.info("Total chunks: %d — generating embeddings...", len(all_chunks))
    texts = [c.text for c in all_chunks]
    embeddings = encoder.encode_documents(texts)

    vs.upsert_chunks(all_chunks, embeddings)
    bm25.build(all_chunks)

    # Persist BM25 index
    bm25_path = Path("data/bm25_index.pkl")
    bm25.save(bm25_path)

    logger.info("✅ Indexed %d chunks. Vector store size: %d", len(all_chunks), vs.count())


def main() -> None:
    parser = argparse.ArgumentParser(description="ScholarRAG Paper Ingestion CLI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="ArXiv search query")
    group.add_argument("--pdf", type=Path, help="Path to a local PDF file")
    parser.add_argument("--max", type=int, default=10, help="Max ArXiv papers to fetch")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/papers"), help="PDF download directory"
    )
    args = parser.parse_args()

    if args.query:
        ingest_arxiv(args.query, args.max, args.output_dir)
    else:
        if not args.pdf.exists():
            logger.error("PDF not found: %s", args.pdf)
            sys.exit(1)
        ingest_pdf(args.pdf)


if __name__ == "__main__":
    main()
