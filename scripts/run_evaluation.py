#!/usr/bin/env python
"""
CLI script to run RAGAS evaluation on the ScholarRAG pipeline.

Usage:
    python scripts/run_evaluation.py --eval-file data/eval_set.json --output eval_results/results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BACKEND_URL = "http://localhost:8000/api/v1"


def load_eval_set(path: Path) -> list[dict]:
    """Load eval set from JSON. Expected format: [{question, ground_truth}, ...]"""
    with open(path) as f:
        return json.load(f)


def run_pipeline(question: str) -> dict:
    r = httpx.post(
        f"{BACKEND_URL}/query",
        json={"question": question, "top_k": 5, "rerank": True},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="ScholarRAG RAGAS Evaluation")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/eval_set.json"),
        help="Path to evaluation JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/results.json"),
        help="Output path for results",
    )
    args = parser.parse_args()

    if not args.eval_file.exists():
        logger.error("Eval file not found: %s", args.eval_file)
        logger.info("Create a JSON file with format: [{question: ..., ground_truth: ...}]")
        sys.exit(1)

    eval_set = load_eval_set(args.eval_file)
    logger.info("Loaded %d eval samples", len(eval_set))

    questions, answers, contexts, ground_truths = [], [], [], []
    for i, sample in enumerate(eval_set):
        question = sample["question"]
        gt = sample.get("ground_truth", "")

        logger.info("[%d/%d] Querying: %r", i + 1, len(eval_set), question)
        try:
            result = run_pipeline(question)
            answers.append(result["answer"])
            contexts.append([s["snippet"] for s in result["sources"]])
            questions.append(question)
            ground_truths.append(gt)
        except Exception as exc:
            logger.error("Failed query %r: %s", question, exc)

    if not questions:
        logger.error("No successful queries. Exiting.")
        sys.exit(1)

    logger.info("Running RAGAS evaluation on %d samples...", len(questions))
    from backend.evaluation.ragas_evaluator import RAGASEvaluator

    evaluator = RAGASEvaluator()
    summary = evaluator.evaluate(
        questions=questions,
        answers=answers,
        contexts=contexts,
        ground_truths=ground_truths if any(ground_truths) else None,
    )

    evaluator.save_results(summary, str(args.output))

    print("\n" + "=" * 50)
    print("📊 RAGAS Evaluation Results")
    print("=" * 50)
    print(f"  Faithfulness:      {summary.avg_faithfulness:.3f}")
    print(f"  Answer Relevancy:  {summary.avg_answer_relevancy:.3f}")
    print(f"  Context Precision: {summary.avg_context_precision:.3f}")
    print(f"  Context Recall:    {summary.avg_context_recall:.3f}")
    print(f"  Overall Average:   {summary.avg_overall:.3f}")
    print("=" * 50)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
