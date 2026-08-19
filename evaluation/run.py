"""Run the local, reproducible evaluation baseline.

Usage: ``python -m evaluation.run`` or pass
``--dataset path/to/samples.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import (
    keyword_context_precision,
    keyword_context_recall,
    token_overlap_faithfulness,
    token_overlap_relevancy,
)

METRIC_NAMES = (
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def evaluate(samples: list[dict]) -> dict[str, float]:
    """Calculate mean retrieval and answer metrics for labelled samples."""
    if not samples:
        raise ValueError("Evaluation dataset is empty")

    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}

    for sample in samples:
        contexts = sample["contexts"]
        keywords = sample["expected_keywords"]
        answer = sample["answer"]
        question = sample["question"]

        values["context_precision"].append(
            keyword_context_precision(contexts, keywords)
        )
        values["context_recall"].append(
            keyword_context_recall(contexts, keywords)
        )
        values["faithfulness"].append(
            token_overlap_faithfulness(answer, contexts)
        )
        values["answer_relevancy"].append(
            token_overlap_relevancy(answer, question)
        )

    return {name: sum(scores) / len(scores) for name, scores in values.items()}


def main() -> None:
    default_dataset = Path(__file__).with_name("sample_dataset.json")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
    )
    args = parser.parse_args()

    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    results = evaluate(samples)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
