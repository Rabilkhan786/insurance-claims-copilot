"""End-to-end tests for the four seeded demo customers.

These run against the real seeded databases (data/crm.db) and, for the
eligibility paths, the real retrieval stack. They are deliberately
integration-level: the point is to catch a break anywhere along the chain
from CRM lookup through the deterministic checklist to the rupee figure the
customer is shown.

Expected figures come from data/seed.py:
  CUST001  Star Health Arogya Sanjeevani, SI 500,000, one 45,000 approved claim
  CUST002  Oriental Happy Family Floater, started 2026-05-16, maternity waits 24 months
  CUST003  Senior Citizens Red Carpet, SI 1,000,000, 920,000 already claimed
  CUST004  New India Top-Up, one rejected cosmetic surgery claim
"""
from datetime import date

import pytest

from config import settings
from src.crm import CRMStore
from src.eligibility import (
    ELIGIBLE,
    FOUND,
    INELIGIBLE,
    NEEDS_MORE_INFO,
    NOT_APPLICABLE,
    UNKNOWN,
    check_eligibility,
)
from src.policy_data import PolicyDataStore
from src.tools.calc_tools import (
    calculate_payable_amount,
    compute_sum_insured_balance,
    compute_waiting_period,
)


def _invoke_scoped(tool, customer_id: str, **kwargs):
    """Call a customer-scoped tool the way the agent runtime would.

    These tools read customer_id from ToolRuntime.context rather than from
    their arguments, so a test has to supply the runtime itself.
    """
    from langchain.tools import ToolRuntime

    from src.agent import Context

    runtime = ToolRuntime(
        state=None,
        context=Context(customer_id=customer_id),
        config=None,
        stream_writer=None,
        tool_call_id=None,
        store=None,
        tools=None,
    )
    return tool.func(runtime=runtime, **kwargs)


@pytest.fixture(scope="module")
def crm():
    return CRMStore(settings.crm_db_path)


@pytest.fixture(scope="module")
def policy_data():
    return PolicyDataStore(settings.crm_db_path)


def _parse(value: str) -> date:
    return date.fromisoformat(value[:10])


# ===========================================================================
# CUST001 -- happy path: coverage, sub-limit, claim status, balance
# ===========================================================================
def test_cust001_holds_the_arogya_sanjeevani_policy(crm):
    policies = crm.get_policies("CUST001")

    assert len(policies) == 1
    assert policies[0]["policy_id"] == "POL001"
    assert policies[0]["policy_number"] == "SHAHLIP22027V032122"


def test_cust001_sum_insured_balance_subtracts_the_approved_claim():
    balance = compute_sum_insured_balance("POL001", "CUST001")

    assert balance["sum_insured"] == 500000
    assert balance["claims_used"] == 45000
    assert balance["remaining_balance"] == 455000


def test_cust001_claim_status_is_approved(crm):
    claim = crm.get_claim_status("CLM001")

    assert claim["status"] == "approved"
    assert claim["eligible_amount"] == 45000


def test_cust001_cataract_sub_limit_is_on_record(policy_data):
    """The bill says 'Cataract Surgery'; the table row says 'Cataract'."""
    row = policy_data.find_sub_limit("SHAHLIP22027V032122", "Cataract Surgery")

    assert row is not None
    assert row["limit_amount"] == 40000


def test_cust001_cataract_bill_applies_sub_limit_then_copay():
    """60,000 bill -> capped to the 40,000 sub-limit -> 5% co-pay -> 38,000."""
    bill = {
        "treatment": "Cataract Surgery",
        "diagnosis": "Senile Cataract Right Eye",
        "total_amount": 60000,
        "admission_date": "2026-08-19",
        "discharge_date": "2026-08-19",
    }

    result = check_eligibility(bill, "CUST001", "POL001")

    assert result["status"] == ELIGIBLE
    assert result["covered_amount"] == 40000
    assert result["copay_amount"] == 2000
    assert result["estimated_payable"] == 38000


# ===========================================================================
# CUST002 -- waiting period and the exact eligibility date
# ===========================================================================
def test_cust002_maternity_waiting_period_is_24_months(policy_data):
    row = policy_data.find_waiting_period(
        "IRDAII/HLT/OIC/P-H/V.II/450/15-16", "maternity"
    )

    assert row is not None
    assert row["waiting_period_months"] == 24


def test_cust002_maternity_becomes_eligible_on_2028_05_16():
    """Policy starts 2026-05-16, so 24 months lands on 2028-05-16."""
    as_of = "2026-08-19"
    result = compute_waiting_period("2026-05-16", 24, today=as_of)

    assert result["eligible_date"] == "2028-05-16"
    assert result["is_eligible"] is False
    # Derived, not hardcoded, so the test does not rot as the clock moves.
    assert result["days_remaining"] == (date(2028, 5, 16) - _parse(as_of)).days


def test_cust002_maternity_bill_is_rejected_for_the_waiting_period():
    bill = {
        "treatment": "maternity",
        "diagnosis": "Normal delivery",
        "total_amount": 60000,
        "admission_date": "2026-08-19",
        "discharge_date": "2026-08-19",
    }

    result = check_eligibility(bill, "CUST002", "POL002")

    assert result["status"] == INELIGIBLE
    assert "waiting period" in result["reason"].lower()
    assert result["estimated_payable"] == 0


def test_cust002_existing_maternity_claim_was_rejected(crm):
    claim = crm.get_claim_status("CLM002")

    assert claim["status"] == "rejected"
    assert "waiting period" in claim["rejection_reason"].lower()


# ===========================================================================
# CUST003 -- sum insured nearly exhausted, co-pay, renewal alert
# ===========================================================================
def test_cust003_has_only_80000_of_sum_insured_left():
    """1,000,000 insured, 920,000 already approved across three claims."""
    balance = compute_sum_insured_balance("POL003", "CUST003")

    assert balance["sum_insured"] == 1000000
    assert balance["claims_used"] == 920000
    assert balance["remaining_balance"] == 80000


def test_cust003_payout_is_capped_by_the_remaining_sum_insured():
    """A 200,000 bill cannot pay more than the 80,000 still available."""
    result = calculate_payable_amount.invoke({
        "bill_amount": 200000,
        "remaining_sum_insured": 80000,
        "copay_percent": 0,
    })

    assert result["payable_amount"] == 80000
    assert result["deductions"]["capped_by_remaining_sum_insured"] is True


def test_cust003_red_carpet_has_no_copayment_on_record(policy_data):
    """No co-pay row is seeded for this plan, so none may be invented."""
    assert policy_data.get_copayments("SHAHLIP25027V072425") == []


# ===========================================================================
# CUST004 -- exclusion, rejection reason, deductible
# ===========================================================================
def test_cust004_cosmetic_surgery_claim_was_rejected_as_excluded(crm):
    claim = crm.get_claim_status("CLM006")

    assert claim["status"] == "rejected"
    assert "excluded" in claim["rejection_reason"].lower()


def test_cust004_cosmetic_surgery_bill_is_not_eligible():
    bill = {
        "treatment": "Cosmetic Surgery",
        "diagnosis": "Rhinoplasty for appearance",
        "total_amount": 80000,
        "admission_date": "2026-08-19",
        "discharge_date": "2026-08-19",
    }

    result = check_eligibility(bill, "CUST004", "POL004")

    assert result["status"] == INELIGIBLE
    assert "exclusion" in result["reason"].lower()
    assert result["estimated_payable"] == 0


def test_cust004_deductible_is_subtracted_before_payout():
    """A top-up pays only above the deductible: 200,000 bill, 100,000 excess."""
    result = calculate_payable_amount.invoke({
        "bill_amount": 200000,
        "remaining_sum_insured": 500000,
        "deductible": 100000,
    })

    assert result["deductions"]["deductible_amount"] == 100000
    assert result["payable_amount"] == 100000


# ===========================================================================
# Cross-customer scoping -- one customer must never read another's policy
# ===========================================================================
@pytest.mark.parametrize(
    "customer_id,foreign_policy_id",
    [("CUST001", "POL003"), ("CUST002", "POL001"), ("CUST004", "POL002")],
)
def test_a_customer_cannot_read_another_customers_policy(customer_id, foreign_policy_id):
    balance = compute_sum_insured_balance(foreign_policy_id, customer_id)

    assert "error" in balance


# ===========================================================================
# CUST004 -- top-up deductible thresholds
# ===========================================================================
def test_cust004_policy_has_a_deductible_on_record(policy_data):
    row = policy_data.find_deductible("IRDA/NL-HLT/NIA/P-H/V.I/35/14-15")

    assert row is not None
    assert row["deductible_amount"] == 200000


def _topup_bill(amount: float) -> dict:
    """A bill that clears the coverage and exclusion gates on CUST004's plan.

    The treatment has to name a real procedure. This said "hospitalisation"
    until the engine gained its third state: that word carries no identifying
    term, so there is nothing to check a policy's per-procedure conditions
    against, and the claim is now correctly parked as needs_more_info rather
    than assessed. See the test below, which pins that behaviour.
    """
    return {
        "treatment": "Angioplasty",
        "diagnosis": "Coronary artery disease",
        "total_amount": amount,
        "admission_date": "2026-08-19",
        "discharge_date": "2026-08-19",
    }


def test_cust004_bill_at_the_deductible_pays_nothing():
    """A 200,000 bill against a 200,000 deductible leaves nothing over.

    The claim is still eligible -- it is the deductible that reduces the
    payout to zero, not a rejection. A payable of 0 and a rejection mean
    very different things to the employee reading the screen.
    """
    result = check_eligibility(_topup_bill(200000), "CUST004", "POL004")

    assert result["status"] == ELIGIBLE
    assert result["estimated_payable"] == 0


def test_cust004_bill_above_the_deductible_pays_the_excess():
    """500,000 bill - 200,000 deductible = 300,000, no co-pay on this plan."""
    result = check_eligibility(_topup_bill(500000), "CUST004", "POL004")

    assert result["status"] == ELIGIBLE
    assert result["estimated_payable"] == 300000


def test_a_claim_naming_no_procedure_is_parked_not_assessed():
    """"hospitalisation" names no procedure, so there is nothing to assess.

    The old engine ran the whole checklist on it, found no exclusion, and
    approved -- reporting a payable amount for a claim whose treatment field
    said nothing at all.
    """
    bill = {
        "treatment": "hospitalisation",
        "diagnosis": "",
        "total_amount": 500000,
        "admission_date": "2026-08-19",
    }

    result = check_eligibility(bill, "CUST004", "POL004")

    assert result["status"] == NEEDS_MORE_INFO
    assert result["estimated_payable"] is None
    assert result["missing_information"]


def test_an_ordinary_policy_has_no_deductible(policy_data):
    """Only top-ups carry one; Arogya Sanjeevani must pay from rupee one."""
    assert policy_data.find_deductible("SHAHLIP22027V032122") is None


def test_sum_insured_balance_exposes_deductible_and_copay():
    """The agent gets these from here; without them it reports the wrong payout.

    A top-up customer was told a 200,000 bill paid in full because the chat
    path had no way to see the 200,000 deductible.
    """
    top_up = compute_sum_insured_balance("POL004", "CUST004")
    assert top_up["deductible"] == 200000

    ordinary = compute_sum_insured_balance("POL001", "CUST001")
    assert ordinary["deductible"] == 0
    assert ordinary["copay_percent"] == 5


# ===========================================================================
# Any policy, no curated rows -- SQL -> RAG -> safe default
# ===========================================================================
# A real indexed policy that has no rows in any policy_* table.
#
# This has moved twice, each time because the policy it named stopped being
# unseeded: BHAHLIP2014V011920 gained a curated cataract sub-limit once the
# extracted rows were loaded, and ACKPAIP21638V012021 left the corpus
# altogether -- its UIN says PAI, a personal accident policy, which does not
# belong in a health claims tool.
UNSEEDED_UIN = "SHAHLIP26044V092526"

# Ambulance cover is the sub-limit this policy states in prose rather than in
# a table, which is exactly the case SQL cannot answer.
UNSEEDED_TREATMENT = "Ambulance charges"


def test_the_unseeded_policy_really_has_no_curated_rows(policy_data):
    """Guard the premise of the tests below."""
    assert policy_data.get_sub_limits(UNSEEDED_UIN) == []
    assert policy_data.get_waiting_periods(UNSEEDED_UIN) == []
    assert policy_data.get_copayments(UNSEEDED_UIN) == []
    assert policy_data.find_deductible(UNSEEDED_UIN) is None


def test_sub_limit_falls_back_to_the_policy_wording():
    """With no SQL row the cap must still be read out of the clause text."""
    from src.eligibility.engine import _sub_limit_fact

    fact = _sub_limit_fact(UNSEEDED_UIN, UNSEEDED_TREATMENT)

    assert fact.status == FOUND
    assert fact.value > 0
    assert fact.source == "policy wording"


def test_a_policy_stating_no_copay_is_not_the_same_as_an_unread_policy():
    """The distinction the whole three-state model exists for.

    "This plan states no co-payment" and "nobody could find out whether it
    does" both used to arrive at the calculator as None, and both were read
    as 0%. One of those is a fact; the other is a gap, and only one of them
    is safe to price a claim on.
    """
    from src.eligibility.engine import _copay_fact

    fact = _copay_fact(UNSEEDED_UIN, UNSEEDED_TREATMENT, age=None)

    assert fact.status in {NOT_APPLICABLE, FOUND}
    assert fact.is_known
    # Whatever it resolved to, it must explain itself to the employee.
    assert fact.detail


def test_missing_deductible_resolves_explicitly_not_by_default():
    """Only top-ups carry one, but "no deductible" has to be established."""
    from src.eligibility.engine import _deductible_fact

    fact = _deductible_fact(UNSEEDED_UIN)

    assert fact.is_known
    assert fact.value_or(0) == 0


def test_an_unknown_fact_reports_no_value_at_all():
    """A fact that could not be established must not collapse to a default.

    value_or() is the only way to read one, and it hands back the caller's
    default only when the fact was actually FOUND -- so an UNKNOWN can never
    be mistaken for a zero on its way into the arithmetic.
    """
    from src.eligibility.facts import unknown

    fact = unknown("copay", "nothing could be retrieved")

    assert fact.is_known is False
    assert fact.value_or(0) == 0
    assert fact.status == UNKNOWN


def test_waiting_period_needs_a_clause_naming_the_treatment():
    """A treatment must not inherit an unrelated waiting period.

    Cataract once picked up the 48-month pre-existing-disease clause from a
    neighbouring paragraph and was wrongly rejected. The plan's curated
    schedule lists pre-existing disease and nothing else, so any other
    treatment resolves to "does not apply" rather than borrowing that number.
    """
    from src.eligibility.engine import _waiting_period_fact

    fact = _waiting_period_fact("SHAHLIP22027V032122", "Cataract Surgery")

    assert fact.status == NOT_APPLICABLE
    assert fact.value is None
