"""
Quality gate for generated answers.
Performs lightweight checks before returning answers to users.
Full RAGAS evaluation is in the evaluation module.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class QualityResult:
    """Result of the quality gate check."""

    passed: bool
    score: float          # 0.0 – 1.0
    reason: str
    checks: dict[str, bool]


class QualityChecker:
    """
    Rule-based quality gate applied before answer delivery.

    Checks:
    1. Answer is not empty
    2. Answer does not start with a refusal ("I cannot...")
    3. Answer contains at least one citation marker [Source N]
    4. Answer length is reasonable (> 50 chars)
    """

    MIN_LENGTH = 50
    CITATION_PATTERN = re.compile(r"\[Source \d+\]")
    REFUSAL_PHRASES = [
        "i cannot find",
        "i don't have enough",
        "not enough information",
        "no relevant information",
    ]

    def check(self, answer: str, question: str) -> QualityResult:  # noqa: ARG002
        checks: dict[str, bool] = {}

        # 1. Non-empty
        checks["non_empty"] = bool(answer.strip())

        # 2. Sufficient length
        checks["min_length"] = len(answer.strip()) >= self.MIN_LENGTH

        # 3. Has citation
        checks["has_citation"] = bool(self.CITATION_PATTERN.search(answer))

        # 4. Not a refusal (soft check — refusals can be valid)
        lower = answer.lower()
        checks["not_refusal"] = not any(p in lower for p in self.REFUSAL_PHRASES)

        # Score = fraction of checks passed
        n_passed = sum(checks.values())
        score = n_passed / len(checks)

        # Gate logic: must pass non_empty + min_length
        passed = checks["non_empty"] and checks["min_length"]

        reason = "All checks passed." if passed else self._build_reason(checks)

        result = QualityResult(passed=passed, score=score, reason=reason, checks=checks)
        logger.info(
            "Quality check: passed=%s, score=%.2f, reason=%s",
            passed, score, reason,
        )
        return result

    @staticmethod
    def _build_reason(checks: dict[str, bool]) -> str:
        failed = [name for name, ok in checks.items() if not ok]
        return f"Failed checks: {', '.join(failed)}"
