"""Tests for the RAG evaluation dataset and its runner.

The claims dataset has its own file (test_claims_evaluation.py) because it
tests decisions, not retrieval. This one only guards the shape of the
questions RAGAS scores.
"""
import json
from collections import Counter
from pathlib import Path

EVALUATION_DIR = Path(__file__).resolve().parents[1] / "evaluation"
DATASET_PATH = EVALUATION_DIR / "rag_dataset.json"
REQUIRED_FIELDS = {"question", "ground_truth", "context_uin", "context_page", "topic"}

# One per kind of clause the eligibility engine actually reads out of a
# policy. Comparison questions were dropped: comparing two policies is not
# something this product does, so measuring it measured nothing.
EXPECTED_TOPICS = {
    "waiting_period": 2,
    "coverage": 2,
    "exclusion": 2,
    "sub_limit": 2,
    "copay_deductible": 2,
}


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_dataset_has_two_questions_per_clause_type():
    dataset = _load_dataset()

    assert Counter(entry["topic"] for entry in dataset) == EXPECTED_TOPICS


def test_every_entry_has_the_required_fields():
    for entry in _load_dataset():
        assert REQUIRED_FIELDS.issubset(entry.keys())


def test_no_entry_has_an_empty_ground_truth():
    assert all(entry["ground_truth"].strip() for entry in _load_dataset())


def test_every_topic_is_one_the_retrieval_router_handles():
    """A topic the runner cannot route falls through to a generic search.

    That silently measures a path the product never takes, which is how the
    old "comparison" topic scored zero on questions whose clause was indexed
    and reachable all along.
    """
    from evaluation.run_ragas import EVAL_TOPICS

    assert {entry["topic"] for entry in _load_dataset()} <= set(EVAL_TOPICS)


def test_ragas_runner_loads_the_dataset_without_errors():
    from evaluation.run_ragas import load_dataset

    assert len(load_dataset()) == sum(EXPECTED_TOPICS.values())


def test_claims_dataset_is_the_main_benchmark_and_is_present():
    """The copilot's own task is scored by claims_dataset.json, not by RAGAS."""
    claims = json.loads(
        (EVALUATION_DIR / "claims_dataset.json").read_text(encoding="utf-8")
    )

    assert len(claims) >= 12
