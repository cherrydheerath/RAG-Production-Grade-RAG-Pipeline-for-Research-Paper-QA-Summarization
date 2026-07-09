"""
Document loader using Unstructured.io for robust PDF parsing.
Falls back to pypdf for simple cases.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from backend.core.exceptions import IngestionError

logger = logging.getLogger(__name__)


@dataclass
class RawDocument:
    """Represents a parsed document before chunking."""

    source: str                          # file path or arxiv_id
    text: str                            # full extracted text
    metadata: dict = field(default_factory=dict)


class DocumentLoader:
    """
    Loads PDFs using Unstructured.io (preferred) with pypdf fallback.

    Unstructured handles:
    - Multi-column layouts
    - Tables
    - Headers / footers
    - Figures (skipped)
    """

    def load_pdf(self, pdf_path: Path, metadata: dict | None = None) -> RawDocument:
        """Load a single PDF file and return a RawDocument."""
        if not pdf_path.exists():
            raise IngestionError(f"PDF not found: {pdf_path}")

        logger.info("Loading PDF: %s", pdf_path.name)

        try:
            text = self._load_with_unstructured(pdf_path)
        except Exception as exc:
            logger.warning(
                "Unstructured failed (%s), falling back to pypdf", exc
            )
            text = self._load_with_pypdf(pdf_path)

        meta = metadata or {}
        meta.setdefault("source", str(pdf_path))
        meta.setdefault("filename", pdf_path.name)

        doc = RawDocument(source=str(pdf_path), text=text, metadata=meta)
        logger.info(
            "Loaded '%s': %d characters", pdf_path.name, len(doc.text)
        )
        return doc

    def load_directory(
        self, directory: Path, metadata_map: dict[str, dict] | None = None
    ) -> list[RawDocument]:
        """Load all PDFs in a directory."""
        pdfs = list(directory.glob("*.pdf"))
        if not pdfs:
            logger.warning("No PDFs found in %s", directory)
            return []

        logger.info("Loading %d PDFs from %s", len(pdfs), directory)
        docs: list[RawDocument] = []
        for pdf in pdfs:
            meta = (metadata_map or {}).get(pdf.name, {})
            try:
                docs.append(self.load_pdf(pdf, meta))
            except IngestionError as exc:
                logger.error("Skipping %s: %s", pdf.name, exc)

        return docs

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _load_with_unstructured(pdf_path: Path) -> str:
        """Use Unstructured.io for high-quality extraction."""
        from unstructured.partition.pdf import partition_pdf  # type: ignore

        elements = partition_pdf(filename=str(pdf_path))
        # Filter out headers/footers/page numbers; keep narrative text
        kept = [
            str(el)
            for el in elements
            if el.category not in {"Header", "Footer", "PageNumber"}
        ]
        return "\n\n".join(kept)

    @staticmethod
    def _load_with_pypdf(pdf_path: Path) -> str:
        """Simple fallback using pypdf."""
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text.strip())
        return "\n\n".join(p for p in pages if p)
