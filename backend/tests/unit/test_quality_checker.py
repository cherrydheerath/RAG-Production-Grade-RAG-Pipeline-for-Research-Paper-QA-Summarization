"""Unit tests for QualityChecker."""
from __future__ import annotations

import pytest

from backend.generation.quality_checker import QualityChecker


checker = QualityChecker()


def test_good_answer_passes():
    answer = (
        "The attention mechanism allows models to focus on relevant parts of the input. [Source 1] "
        "This was introduced by Vaswani et al. [Source 2]"
    )
    result = checker.check(answer, "What is attention?")
    assert result.passed is True
    assert result.score > 0.5


def test_empty_answer_fails():
    result = checker.check("", "What is attention?")
    assert result.passed is False
    assert result.checks["non_empty"] is False


def test_short_answer_fails():
    result = checker.check("Yes.", "Explain transformers.")
    assert result.passed is False
    assert result.checks["min_length"] is False


def test_answer_with_citation():
    answer = "Deep learning enables feature extraction. [Source 1] " * 3
    result = checker.check(answer, "What is deep learning?")
    assert result.checks["has_citation"] is True


def test_refusal_detected():
    answer = (
        "I cannot find sufficient information in the provided papers to answer this question. "
        "Please check if relevant papers are indexed."
    )
    result = checker.check(answer, "What is X?")
    assert result.checks["not_refusal"] is False


def test_score_between_zero_and_one():
    answer = "Some answer with [Source 1] citation here." * 2
    result = checker.check(answer, "Question?")
    assert 0.0 <= result.score <= 1.0
