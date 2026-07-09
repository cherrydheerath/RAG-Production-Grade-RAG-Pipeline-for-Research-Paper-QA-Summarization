"""
LangSmith tracing integration.
Wraps key pipeline steps with LangSmith run tracking.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Generator

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


def configure_langsmith() -> None:
    """Set LangSmith environment variables from settings."""
    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)
        logger.info("LangSmith tracing enabled for project: %s", settings.langchain_project)
    else:
        logger.info("LangSmith tracing disabled")


@contextmanager
def trace_pipeline_step(
    name: str, inputs: dict | None = None
) -> Generator[None, None, None]:
    """
    Context manager that wraps a pipeline step with LangSmith tracing.
    Falls back gracefully if LangSmith is not configured.
    """
    if not settings.langchain_tracing_v2:
        yield
        return

    try:
        from langsmith import traceable  # noqa: F401
        # LangSmith auto-traces when LANGCHAIN_TRACING_V2=true
        logger.debug("LangSmith: starting trace for '%s'", name)
        yield
        logger.debug("LangSmith: completed trace for '%s'", name)
    except ImportError:
        logger.debug("langsmith not installed, skipping trace")
        yield
    except Exception as exc:
        logger.warning("LangSmith trace error (non-fatal): %s", exc)
        yield
