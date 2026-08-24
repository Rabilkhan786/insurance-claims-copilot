"""Tests for the RAGAS evaluation dataset and runner."""
import json
from collections import Counter
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "dataset.json"
REQUIRED_FIELDS = {"question", "ground_truth", "context_uin", "context_page", "topic"}


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_dataset_has_exactly_10_entries():
    dataset = _load_dataset()

    assert len(dataset) == 10


def test_dataset_has_two_questions_per_topic():
    dataset = _load_dataset()

    counts = Counter(entry["topic"] for entry in dataset)

    assert counts == {
        "waiting_period": 2,
        "coverage": 2,
        "exclusion": 2,
        "sub_limit": 2,
        "comparison": 2,
    }


def test_every_entry_has_the_required_fields():
    dataset = _load_dataset()

    for entry in dataset:
        assert REQUIRED_FIELDS.issubset(entry.keys())


def test_no_entry_has_an_empty_ground_truth():
    dataset = _load_dataset()

    assert all(entry["ground_truth"].strip() for entry in dataset)


def test_ragas_runner_loads_the_dataset_without_errors():
    from evaluation.run_ragas import load_dataset

    entries = load_dataset()

    assert len(entries) == 10
