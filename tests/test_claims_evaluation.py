"""Score the copilot on the task it actually does: assessing a claim.

WHY this is pytest and not RAGAS: every number here is deterministic. A
payable amount is either Rs 38,000 or it is wrong, and asking an LLM judge
whether Rs 41,753 is "close enough" would be a worse test than ==. RAGAS
scores the retrieval quality behind these decisions (evaluation/run_ragas.py);
this file scores the decisions themselves.

Each case in evaluation/claims_dataset.json is run through the real engine
against the real seeded databases and the real retrieval stack, then checked
on the two things that matter to a claims employee:

    status          eligible / ineligible / needs_more_info
    payable_amount  to the rupee

Run just this file with:  uv run pytest tests/test_claims_evaluation.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.workflow import _to_bill_data
from src.eligibility import check_eligibility

DATASET_PATH = Path(__file__).resolve().parents[1] / "evaluation" / "claims_dataset.json"

# The scenarios the dataset has to keep covering. If a case is ever deleted,
# this list is what fails rather than the suite quietly getting weaker.
REQUIRED_SCENARIOS = {
    "sub-limit",
    "co-pay",
    "deductible",
    "waiting-period",
    "exclusion",
    "sum insured",
    "expired",
    "mismatch",
    "missing claim information",
    "missing policy evidence",
}


def _load_cases() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


CASES = _load_cases()


# The dataset's claim shape is the employee form's shape, so it is mapped with
# the workflow's own function rather than a copy of it -- a second copy would
# let the evaluation drift away from the path a real submission takes, and pass
# while production broke.


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    """Run every case once and share the results across the tests below.

    Module-scoped because a case that needs retrieval costs a few seconds,
    and three tests read each result.
    """
    return {
        case["case_id"]: check_eligibility(
            _to_bill_data(case["claim"]), case["customer_id"], case["policy_id"]
        )
        for case in CASES
    }


# --- the dataset itself -----------------------------------------------------
def test_dataset_covers_every_required_scenario():
    """The benchmark is only as good as the situations it contains."""
    text = " ".join(
        f"{case['scenario']} {case['case_id']}" for case in CASES
    ).lower()

    missing = [name for name in REQUIRED_SCENARIOS if name not in text]

    assert not missing, f"claims_dataset.json no longer covers: {missing}"


def test_every_case_has_a_complete_expectation():
    for case in CASES:
        expected = case["expected"]
        assert expected["status"] in {"eligible", "ineligible", "needs_more_info"}
        assert expected["reason"].strip()
        # payable_amount is allowed to be null, but only for needs_more_info:
        # an assessed claim always has a figure, even when that figure is 0.
        if expected["payable_amount"] is None:
            assert expected["status"] == "needs_more_info"


# --- the decisions ----------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_status_matches_expectation(case, results):
    actual = results[case["case_id"]]

    assert actual["status"] == case["expected"]["status"], (
        f"{case['case_id']}: expected {case['expected']['status']}, "
        f"got {actual['status']} -- {actual.get('reason')}"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_payable_amount_matches_expectation(case, results):
    actual = results[case["case_id"]]
    expected = case["expected"]["payable_amount"]

    if expected is None:
        assert actual["estimated_payable"] is None, (
            f"{case['case_id']}: no amount should be offered when a fact the "
            f"decision depends on is missing"
        )
        return

    assert actual["estimated_payable"] == pytest.approx(expected, abs=0.01), (
        f"{case['case_id']}: expected Rs {expected}, "
        f"got Rs {actual['estimated_payable']} -- {actual.get('reason')}"
    )


# --- the guarantees that hold across every case -----------------------------
@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_needs_more_info_always_names_what_is_missing(case, results):
    """A claim parked for more information must say what to go and find."""
    actual = results[case["case_id"]]
    if actual["status"] != "needs_more_info":
        return

    assert actual["missing_information"], (
        f"{case['case_id']}: parked as needs_more_info without naming anything"
    )


@pytest.mark.parametrize("case", CASES, ids=[c["case_id"] for c in CASES])
def test_no_fact_is_silently_assumed(case, results):
    """Any unestablished fact must stop the claim, never reach the maths.

    This is the regression guard for the behaviour this engine was rewritten
    to remove: an empty lookup being read as zero, and the employee being
    shown a confident payable amount built on it.
    """
    actual = results[case["case_id"]]
    unknown = [
        name
        for name, fact in (actual.get("facts") or {}).items()
        if fact["status"] == "unknown"
    ]

    if unknown:
        assert actual["status"] == "needs_more_info", (
            f"{case['case_id']}: {unknown} were never established, but the "
            f"engine still returned {actual['status']} with a payable of "
            f"{actual['estimated_payable']}"
        )
