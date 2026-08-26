"""Tests for the typed recommendation and the audit row built from it.

These run without the network: a ClaimRecommendation is assembled from an
engine result, so it can be tested against a hand-written one.
"""
from __future__ import annotations

import pytest

from src.agent.recommendation import ClaimRecommendation
from src.decisions import DecisionStore
from src.eligibility import ELIGIBLE, INELIGIBLE, NEEDS_MORE_INFO


def _engine_result(**overrides) -> dict:
    """An eligible engine result, in the shape check_eligibility returns."""
    base = {
        "status": ELIGIBLE,
        "bill_amount": 60000,
        "covered_amount": 40000,
        "copay_amount": 2000,
        "estimated_payable": 38000,
        "reason": "Cataract is capped at Rs 40,000, then a 5% co-pay applies.",
        "evidence": [
            {
                "text": "Cataract is limited to Rs 40,000 per eye.",
                "uin": "SHAHLIP22027V032122",
                "insurer": "Star Health",
                "page": 8,
            }
        ],
        "missing_information": [],
        "facts": {
            "sub_limit": {
                "name": "sub_limit",
                "status": "found",
                "value": 40000,
                "source": "policy records",
                "detail": "Sub-limit of Rs 40,000 on record.",
            }
        },
        "deductions": {"sub_limit_reduction": 20000, "copay_amount": 2000},
    }
    return {**base, **overrides}


# --- the engine stays authoritative ------------------------------------------
def test_payable_amount_comes_from_the_engine_not_the_prose():
    """The model's text cannot move the figure, whatever it claims.

    A model once wrote Rs 41,753 into its reasoning where the engine had
    calculated Rs 38,000. The recommendation is assembled from the engine, so
    the prose is the only thing the model contributes.
    """
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(),
        reasoning="Recommend: approve. The payable amount is Rs 41,753.",
    )

    assert recommendation.payable_amount == 38000


@pytest.mark.parametrize(
    "engine_status,expected",
    [
        (ELIGIBLE, "approve"),
        (INELIGIBLE, "reject"),
        (NEEDS_MORE_INFO, "needs_more_info"),
    ],
)
def test_every_engine_state_maps_to_a_recommendation(engine_status, expected):
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(status=engine_status), reasoning=""
    )

    assert recommendation.status == expected


def test_an_unrecognised_engine_status_is_never_read_as_approve():
    """The safe direction for a status nobody planned for is "ask a human"."""
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(status="something_new"), reasoning=""
    )

    assert recommendation.status == "needs_more_info"


def test_needs_more_info_carries_no_payable_amount():
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(
            status=NEEDS_MORE_INFO,
            estimated_payable=None,
            missing_information=["The co-payment could not be established."],
        ),
        reasoning="",
    )

    assert recommendation.payable_amount is None
    assert recommendation.missing_information


# --- citations ---------------------------------------------------------------
def test_evidence_keeps_everything_a_citation_needs():
    recommendation = ClaimRecommendation.from_engine(_engine_result(), reasoning="")

    evidence = recommendation.evidence[0]

    assert evidence.is_citable
    assert evidence.citation() == (
        "[Source: Star Health, UIN: SHAHLIP22027V032122, Page 8]"
    )


def test_a_clause_without_a_uin_or_page_is_not_citable():
    """The prompt forbids stating a fact from one; this is what marks it."""
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(
            evidence=[{"text": "Some clause.", "uin": None, "insurer": "X", "page": None}]
        ),
        reasoning="",
    )

    assert recommendation.evidence[0].is_citable is False


def test_evidence_without_text_is_dropped():
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(evidence=[{"text": "", "uin": "U", "insurer": "X", "page": 1}]),
        reasoning="",
    )

    assert recommendation.evidence == []


# --- the audit trail ---------------------------------------------------------
def test_audit_row_keeps_the_recommendation_and_the_decision_apart(tmp_path):
    """An edited amount must not overwrite what the AI actually proposed.

    This is the whole point of the table: an audit asks "did the human agree
    with the machine", and storing only the outcome destroys that.
    """
    store = DecisionStore(tmp_path / "crm.db")
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(), reasoning="Recommend: approve."
    ).model_dump()

    decision_id = store.record(
        claim_id="CLM-TEST",
        customer_id="CUST001",
        policy_id="POL001",
        recommendation=recommendation,
        employee_decision="edit",
        employee_payable_amount=25000,
        employee_edits="Hospital billed a second eye separately.",
        decided_by="emp.demo",
    )

    row = store.get(decision_id)

    assert row["ai_decision"] == "approve"
    assert row["ai_payable_amount"] == 38000
    assert row["employee_decision"] == "edit"
    assert row["employee_payable_amount"] == 25000
    # Still the AI's original figure, untouched by the edit.
    assert row["ai_recommendation"]["payable_amount"] == 38000
    assert row["agreed"] == 0


def test_agreement_is_only_recorded_when_the_employee_approved(tmp_path):
    store = DecisionStore(tmp_path / "crm.db")
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(), reasoning=""
    ).model_dump()

    store.record(
        claim_id="CLM-A", customer_id="C", recommendation=recommendation,
        employee_decision="approve", decided_by="emp.demo",
    )
    store.record(
        claim_id="CLM-B", customer_id="C", recommendation=recommendation,
        employee_decision="reject", decided_by="emp.demo", override_reason="No.",
    )

    assert store.agreement_rate() == {
        "total": 2, "agreed": 1, "agreement_rate_percent": 50.0,
    }


def test_a_needs_more_info_recommendation_is_recorded_as_such(tmp_path):
    """The audit has to show the copilot declined to call it, not that it rejected."""
    store = DecisionStore(tmp_path / "crm.db")
    recommendation = ClaimRecommendation.from_engine(
        _engine_result(status=NEEDS_MORE_INFO, estimated_payable=None), reasoning=""
    ).model_dump()

    decision_id = store.record(
        claim_id="CLM-C", customer_id="C", recommendation=recommendation,
        employee_decision="approve", employee_payable_amount=38000,
        decided_by="emp.demo",
    )

    assert store.get(decision_id)["ai_decision"] == "needs_more_info"


def test_an_unknown_employee_decision_is_refused(tmp_path):
    store = DecisionStore(tmp_path / "crm.db")

    with pytest.raises(ValueError):
        store.record(
            claim_id="CLM-D", customer_id="C", recommendation={"status": "approve"},
            employee_decision="maybe", decided_by="emp.demo",
        )
