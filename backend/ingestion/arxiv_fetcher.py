"""
ArXiv paper fetcher using the arxiv Python SDK.
Fetches paper metadata + downloads PDFs for ingestion.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import arxiv

from backend.core.config import get_settings
from backend.core.exceptions import IngestionError

logger = logging.getLogger(__name__)
settings = get_settings()


class ArXivFetcher:
    """Fetch papers from ArXiv API and optionally download PDFs."""

    def __init__(self, max_results: int | None = None) -> None:
        self.max_results = max_results or settings.arxiv_max_results
        self.client = arxiv.Client(
            page_size=min(self.max_results, 100),
            delay_seconds=3.0,
            num_retries=3,
        )

    def search(self, query: str, max_results: int | None = None) -> list[dict]:
        """
        Search ArXiv and return list of paper metadata dicts.

        Args:
            query: ArXiv search query string
            max_results: Override default max results

        Returns:
            List of paper metadata dicts
        """
        n = max_results or self.max_results
        logger.info("Searching ArXiv: query=%r, max_results=%d", query, n)

        search = arxiv.Search(
            query=query,
            max_results=n,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        papers: list[dict] = []
        try:
            for result in self.client.results(search):
                papers.append(self._result_to_dict(result))
                logger.debug("Fetched paper: %s", result.title)
        except Exception as exc:
            raise IngestionError(f"ArXiv search failed: {exc}") from exc

        logger.info("Fetched %d papers from ArXiv", len(papers))
        return papers

    def download_pdf(self, paper: dict, output_dir: Path) -> Path:
        """
        Download a single paper PDF to output_dir.

        Returns:
            Path to downloaded PDF
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        arxiv_id = paper["arxiv_id"]
        pdf_path = output_dir / f"{arxiv_id.replace('/', '_')}.pdf"

        if pdf_path.exists():
            logger.debug("PDF already exists: %s", pdf_path)
            return pdf_path

        try:
            result = next(self.client.results(arxiv.Search(id_list=[arxiv_id])))
            result.download_pdf(dirpath=str(output_dir), filename=pdf_path.name)
            logger.info("Downloaded PDF: %s", pdf_path.name)
            time.sleep(1)  # be polite to ArXiv
        except Exception as exc:
            raise IngestionError(f"PDF download failed for {arxiv_id}: {exc}") from exc

        return pdf_path

    def download_batch(
        self, papers: list[dict], output_dir: Path, limit: int | None = None
    ) -> list[Path]:
        """Download PDFs for a list of papers."""
        results: list[Path] = []
        items = papers[:limit] if limit else papers
        for paper in items:
            try:
                path = self.download_pdf(paper, output_dir)
                results.append(path)
            except IngestionError as exc:
                logger.warning("Skipping %s: %s", paper.get("arxiv_id"), exc)
        return results

    @staticmethod
    def _result_to_dict(result: arxiv.Result) -> dict:
        return {
            "arxiv_id": result.entry_id.split("/")[-1],
            "title": result.title,
            "authors": [a.name for a in result.authors],
            "abstract": result.summary,
            "categories": result.categories,
            "published": result.published.isoformat() if result.published else "",
            "updated": result.updated.isoformat() if result.updated else "",
            "pdf_url": result.pdf_url,
            "doi": result.doi or "",
        }
