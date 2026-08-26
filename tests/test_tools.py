"""Tests for the agent tools: CRM and calculation functions.

Each test that touches a database points the relevant module's singleton
at an isolated tmp_path store via monkeypatch, so tests never depend on
(or pollute) the real seeded data.
"""
from datetime import datetime, timedelta

from langchain.tools import ToolRuntime

import src.tools.calc_tools as calc_tools
import src.tools.crm_tools as crm_tools
from src.agent import Context
from src.crm import CRMStore


def _invoke_scoped(tool, customer_id: str, **kwargs):
    """Call a customer-scoped tool the way the agent runtime would.

    These tools take customer_id from ToolRuntime.context rather than from
    their arguments, so a test has to supply the runtime itself.
    """
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


# --- CRM tools --------------------------------------------------------------
def test_get_customer_returns_an_error_for_unknown_customer_id(tmp_path, monkeypatch):
    """The tool returns a structured error, not None — the model reads the
    message and tells the customer, instead of seeing an empty result."""
    store = CRMStore(tmp_path / "crm.db")
    monkeypatch.setattr(crm_tools, "_store", store)

    assert _invoke_scoped(crm_tools.get_customer, "NOPE") == {
        "error": "Customer not found."
    }


def test_get_policies_returns_empty_list_for_customer_with_no_policies(tmp_path, monkeypatch):
    store = CRMStore(tmp_path / "crm.db")
    store.add_customer("A1", "Alice")
    monkeypatch.setattr(crm_tools, "_store", store)

    assert _invoke_scoped(crm_tools.get_policies, "A1") == []


def test_customer_scoped_tools_hide_customer_id_from_the_model():
    """The LLM must not be able to supply (or be asked for) a customer_id."""
    for tool in (crm_tools.get_customer, crm_tools.get_policies, crm_tools.get_claims):
        assert "customer_id" not in tool.args


# --- calc_tools: waiting period ----------------------------------------------
def test_waiting_period_tracker_not_eligible_six_months_into_two_year_wait():
    result = calc_tools.compute_waiting_period(
        policy_start_date="2026-02-16",
        waiting_period_months=24,
        today="2026-08-16",
    )

    assert result["is_eligible"] is False


def test_waiting_period_tracker_eligible_after_two_years():
    result = calc_tools.compute_waiting_period(
        policy_start_date="2024-08-16",
        waiting_period_months=24,
        today="2026-08-16",
    )

    assert result["is_eligible"] is True


# --- calc_tools: payable amount -----------------------------------------------
def test_calculate_payable_amount_applies_sub_limit_and_copay():
    result = calc_tools.calculate_payable_amount.invoke({
        "bill_amount": 60000,
        "remaining_sum_insured": 500000,
        "sub_limit": 40000,
        "copay_percent": 5,
    })

    # Sub-limit caps the covered amount to 40000, then 5% copay is deducted.
    assert result["deductions"]["sub_limit_reduction"] == 20000
    assert result["deductions"]["copay_amount"] == 2000.0
    assert result["payable_amount"] == 38000.0


# --- calc_tools: sum insured balance -----------------------------------------
def test_sum_insured_balance_returns_correct_remaining_amount(tmp_path, monkeypatch):
    store = CRMStore(tmp_path / "crm.db")
    store.add_customer("A1", "Alice")
    store.add_policy(
        "POL-A1", "A1", "PN-A1", "individual", "Alice's Policy",
        sum_insured=1000000, start_date="2025-09-01", end_date="2026-08-31",
    )
    store.add_claim(
        "CLM-A1", "A1", "POL-A1", "hospitalization", 400000,
        "2025-10-05", status="approved", eligible_amount=400000,
    )
    store.add_claim(
        "CLM-A2", "A1", "POL-A1", "hospitalization", 350000,
        "2026-01-20", status="approved", eligible_amount=350000,
    )
    # A rejected claim must not count against the balance.
    store.add_claim(
        "CLM-A3", "A1", "POL-A1", "hospitalization", 999999,
        "2026-02-01", status="rejected",
    )
    # The store is reached through a cached accessor, so the accessor is what
    # gets swapped -- setting a module global would leave the cache in place.
    monkeypatch.setattr(calc_tools, "get_crm_store", lambda: store)

    result = calc_tools.compute_sum_insured_balance("POL-A1", "A1")

    assert result["claims_used"] == 750000
    assert result["remaining_balance"] == 250000


# --- policy_data: sub-limit matching (regression) ------------------------------
def _sub_limit_store(tmp_path):
    """A store holding one 'Cataract' sub-limit row, as the real data does."""
    from src.policy_data import PolicyDataStore

    store = PolicyDataStore(tmp_path / "policy.db")
    store.add_sub_limit("UIN1", "Star Health", "Cataract", 40000, page=8)
    store.add_sub_limit("UIN1", "Star Health", "All other major surgeries", 60000, page=8)
    return store


def test_find_sub_limit_matches_a_longer_bill_treatment_name(tmp_path):
    """A bill saying 'Cataract Surgery' must still find the 'Cataract' row.

    The old LIKE only matched when the stored name was the longer string, so
    this returned None and the 40,000 cap was silently skipped.
    """
    store = _sub_limit_store(tmp_path)

    row = store.find_sub_limit("UIN1", "Cataract Surgery")

    assert row is not None
    assert row["limit_amount"] == 40000


def test_find_sub_limit_does_not_match_on_a_generic_word_alone(tmp_path):
    """'surgery' identifies nothing, so it must not hit the cataract row."""
    store = _sub_limit_store(tmp_path)

    assert store.find_sub_limit("UIN1", "surgery") is None


def test_calculate_payable_amount_applies_the_cataract_sub_limit_end_to_end():
    """60,000 bill, 40,000 sub-limit, 5% copay -> 38,000 payable."""
    result = calc_tools.calculate_payable_amount.invoke({
        "bill_amount": 60000,
        "remaining_sum_insured": 500000,
        "sub_limit": 40000,
        "copay_percent": 5,
    })

    assert result["payable_amount"] == 38000.0
