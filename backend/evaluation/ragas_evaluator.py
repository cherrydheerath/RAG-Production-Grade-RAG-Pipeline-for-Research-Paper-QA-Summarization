"""
RAGAS-based evaluation pipeline.
Measures: faithfulness, answer_relevancy, context_precision, context_recall.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """Evaluation result for a single QA pair."""

    question: str
    answer: str
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def average(self) -> float:
        return (
            self.faithfulness
            + self.answer_relevancy
            + self.context_precision
            + self.context_recall
        ) / 4


@dataclass
class EvaluationSummary:
    """Aggregate metrics across eval set."""

    n_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_overall: float
    per_sample: list[RAGASResult] = field(default_factory=list)


class RAGASEvaluator:
    """
    Evaluates the RAG pipeline using RAGAS metrics.

    Usage:
        evaluator = RAGASEvaluator()
        summary = evaluator.evaluate(eval_dataset)
    """

    def __init__(self) -> None:
        self._check_ragas()

    @staticmethod
    def _check_ragas() -> None:
        try:
            import ragas  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ragas is not installed. Run: pip install ragas"
            ) from exc

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> EvaluationSummary:
        """
        Run RAGAS evaluation.

        Args:
            questions: List of user questions
            answers: List of generated answers
            contexts: List of context chunks per question
            ground_truths: Optional ground truth answers for recall

        Returns:
            EvaluationSummary with per-metric aggregates
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        metrics = [faithfulness, answer_relevancy, context_precision]

        data: dict[str, Any] = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            from ragas.metrics import context_recall
            data["ground_truth"] = ground_truths
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation on %d samples...", len(questions))
        result = evaluate(dataset, metrics=metrics)
        df = result.to_pandas()

        per_sample: list[RAGASResult] = []
        for i, row in df.iterrows():
            per_sample.append(
                RAGASResult(
                    question=row.get("question", ""),
                    answer=row.get("answer", ""),
                    faithfulness=float(row.get("faithfulness", 0) or 0),
                    answer_relevancy=float(row.get("answer_relevancy", 0) or 0),
                    context_precision=float(row.get("context_precision", 0) or 0),
                    context_recall=float(row.get("context_recall", 0) or 0),
                )
            )

        summary = EvaluationSummary(
            n_samples=len(per_sample),
            avg_faithfulness=sum(r.faithfulness for r in per_sample) / len(per_sample),
            avg_answer_relevancy=sum(r.answer_relevancy for r in per_sample) / len(per_sample),
            avg_context_precision=sum(r.context_precision for r in per_sample) / len(per_sample),
            avg_context_recall=sum(r.context_recall for r in per_sample) / len(per_sample),
            avg_overall=sum(r.average for r in per_sample) / len(per_sample),
            per_sample=per_sample,
        )

        logger.info(
            "RAGAS results: faithfulness=%.3f, relevancy=%.3f, precision=%.3f, overall=%.3f",
            summary.avg_faithfulness,
            summary.avg_answer_relevancy,
            summary.avg_context_precision,
            summary.avg_overall,
        )
        return summary

    def save_results(self, summary: EvaluationSummary, output_path: str) -> None:
        """Save evaluation results to JSON."""
        import json
        from pathlib import Path

        data = {
            "n_samples": summary.n_samples,
            "avg_faithfulness": summary.avg_faithfulness,
            "avg_answer_relevancy": summary.avg_answer_relevancy,
            "avg_context_precision": summary.avg_context_precision,
            "avg_context_recall": summary.avg_context_recall,
            "avg_overall": summary.avg_overall,
            "per_sample": [
                {
                    "question": r.question,
                    "answer": r.answer,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                }
                for r in summary.per_sample
            ],
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved evaluation results to %s", output_path)
