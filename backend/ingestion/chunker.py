"""
Text chunker using LangChain's RecursiveCharacterTextSplitter.
Produces overlapping chunks with rich metadata for retrieval.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import get_settings
from backend.ingestion.document_loader import RawDocument

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class DocumentChunk:
    """A single text chunk ready for embedding."""

    chunk_id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)
    # position within source document
    chunk_index: int = 0
    total_chunks: int = 0


class TextChunker:
    """
    Splits RawDocuments into overlapping chunks.

    Uses RecursiveCharacterTextSplitter which tries paragraph → sentence →
    word splits to preserve semantic boundaries.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,
        )
        logger.info(
            "TextChunker init: chunk_size=%d, overlap=%d",
            self.chunk_size,
            self.chunk_overlap,
        )

    def chunk_document(self, doc: RawDocument) -> list[DocumentChunk]:
        """Split a single document into chunks."""
        if not doc.text.strip():
            logger.warning("Empty document: %s", doc.source)
            return []

        raw_chunks = self._splitter.create_documents(
            texts=[doc.text],
            metadatas=[doc.metadata],
        )

        chunks: list[DocumentChunk] = []
        total = len(raw_chunks)
        for idx, raw in enumerate(raw_chunks):
            chunk_id = f"{doc.source}::chunk_{idx}"
            meta = {**raw.metadata, "chunk_index": idx, "total_chunks": total}
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=raw.page_content,
                    source=doc.source,
                    metadata=meta,
                    chunk_index=idx,
                    total_chunks=total,
                )
            )

        logger.info(
            "Chunked '%s' → %d chunks (size=%d, overlap=%d)",
            doc.source,
            total,
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def chunk_documents(self, docs: list[RawDocument]) -> list[DocumentChunk]:
        """Chunk a list of documents."""
        all_chunks: list[DocumentChunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(
            "Total chunks produced: %d from %d documents", len(all_chunks), len(docs)
        )
        return all_chunks
