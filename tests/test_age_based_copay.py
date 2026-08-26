"""Tests for age-banded co-pay: the store's filter, and the engine using it.

WHY real data: none of the seeded demo policies (Data/seed.py) have an
age-banded co-pay row, so these tests build their own isolated store rather
than touch the shared demo dataset. The age band used is not invented --
it is Future Generali's "Health Total" policy (UIN
IRDAI/HLT/FGII/P-H/V.I/02/15-16), verified in artifacts/staged_tables.json
against the real indexed PDF, page 3: co-payment rises with age for anyone
first covered under the policy at 60 or older.

    60-64  20%
    65-69  25%
    70-74  30%
    75+    40%

No demo customer holds this policy, so these tests are additive: they do not
change what any of the 15 cases in evaluation/claims_dataset.json compute,
since none of those touch a UIN with an age-banded row.
"""
from __future__ import annotations

import pytest

from src.eligibility.engine import _copay_fact, _customer_age_at_treatment
from src.policy_data import PolicyDataStore
from src.tools.calc_tools import compute_age

FGII_UIN = "IRDAI/HLT/FGII/P-H/V.I/02/15-16"

# Transcribed verbatim from the real clause, not invented.
FGII_AGE_BANDS = [
    {"condition": "First-time covered, age 60 to 64", "copay_percent": 20, "age_min": 60, "age_max": 64},
    {"condition": "First-time covered, age 65 to 69", "copay_percent": 25, "age_min": 65, "age_max": 69},
    {"condition": "First-time covered, age 70 to 74", "copay_percent": 30, "age_min": 70, "age_max": 74},
    {"condition": "First-time covered, age 75 and above", "copay_percent": 40, "age_min": 75, "age_max": None},
]


@pytest.fixture
def fgii_store(tmp_path):
    """An isolated store holding only the real Future Generali age bands."""
    store = PolicyDataStore(tmp_path / "policy.db")
    for row in FGII_AGE_BANDS:
        store.add_copayment(policy_uin=FGII_UIN, insurer="Future Generali", page=3, **row)
    return store


# --- compute_age --------------------------------------------------------
def test_compute_age_before_this_years_birthday_is_one_less():
    """Turns 26 on 2026-06-15; assessed six weeks earlier, still 25."""
    assert compute_age("2000-06-15", "2026-05-01") == 25


def test_compute_age_on_the_birthday_itself_counts_the_new_year():
    assert compute_age("2000-06-15", "2026-06-15") == 26


def test_compute_age_uses_the_treatment_date_not_a_later_one():
    """The same date of birth gives a different age depending on as_of."""
    dob = "1960-01-01"
    assert compute_age(dob, "2020-06-01") == 60
    assert compute_age(dob, "2025-06-01") == 65


# --- PolicyDataStore.get_copayments age filter --------------------------
def test_age_inside_a_band_returns_only_that_row(fgii_store):
    rows = fgii_store.get_copayments(FGII_UIN, age=62)

    assert len(rows) == 1
    assert rows[0]["copay_percent"] == 20


def test_age_outside_every_band_returns_nothing(fgii_store):
    """39 is younger than any band this clause defines -- no row applies."""
    assert fgii_store.get_copayments(FGII_UIN, age=39) == []


def test_age_at_the_open_ended_top_band_matches(fgii_store):
    """age_max=None means "and above" -- 90 must still hit the 75+ row."""
    rows = fgii_store.get_copayments(FGII_UIN, age=90)

    assert len(rows) == 1
    assert rows[0]["copay_percent"] == 40


def test_no_age_given_returns_every_band_unfiltered(fgii_store):
    """age=None is "don't filter", not "filter to nothing"."""
    assert len(fgii_store.get_copayments(FGII_UIN)) == 4


def test_a_row_with_no_age_band_matches_any_age(tmp_path):
    """The existing seeded shape: age_min and age_max both NULL means always."""
    store = PolicyDataStore(tmp_path / "policy.db")
    store.add_copayment(
        policy_uin="SHAHLIP22027V032122", insurer="Star Health",
        condition="all claims", copay_percent=5, age_min=None, age_max=None, page=8,
    )

    for age in (5, 45, 90, None):
        rows = store.get_copayments("SHAHLIP22027V032122", age=age)
        assert len(rows) == 1
        assert rows[0]["copay_percent"] == 5


def test_overlapping_bands_resolve_the_same_way_every_time(tmp_path):
    """The schema does not forbid overlap; behaviour must still be stable.

    No real policy in this corpus has overlapping bands -- this constructs
    one to prove the code path is deterministic, not to claim it is a real
    clause. See _find_applicable_copay's docstring for why this falls
    through to "whichever row the store returns first" rather than a
    "narrowest band wins" rule nothing in the corpus asks for.
    """
    store = PolicyDataStore(tmp_path / "policy.db")
    store.add_copayment(
        policy_uin="TESTUIN", insurer="Test", condition="Band A",
        copay_percent=10, age_min=60, age_max=70, page=1,
    )
    store.add_copayment(
        policy_uin="TESTUIN", insurer="Test", condition="Band B",
        copay_percent=20, age_min=65, age_max=75, page=1,
    )

    first = store.get_copayments("TESTUIN", age=68)
    second = store.get_copayments("TESTUIN", age=68)

    assert len(first) == 2  # both bands genuinely include age 68
    assert first == second  # same query, same database, same order every time


# --- _customer_age_at_treatment ------------------------------------------
def test_age_is_none_without_a_treatment_date(monkeypatch):
    import src.eligibility.engine as engine

    class _Store:
        def get_customer(self, customer_id):
            return {"date_of_birth": "1960-01-01"}

    monkeypatch.setattr(engine, "get_crm_store", lambda: _Store())

    assert _customer_age_at_treatment("CUST-X", None) is None


def test_age_is_none_without_a_date_of_birth_on_file(monkeypatch):
    import src.eligibility.engine as engine

    class _Store:
        def get_customer(self, customer_id):
            return {"date_of_birth": None}

    monkeypatch.setattr(engine, "get_crm_store", lambda: _Store())

    assert _customer_age_at_treatment("CUST-X", "2026-01-01") is None


def test_age_is_computed_when_both_are_on_file(monkeypatch):
    import src.eligibility.engine as engine

    class _Store:
        def get_customer(self, customer_id):
            return {"date_of_birth": "1960-01-01"}

    monkeypatch.setattr(engine, "get_crm_store", lambda: _Store())

    assert _customer_age_at_treatment("CUST-X", "2026-06-01") == 66


# --- _copay_fact: the engine actually using the band ---------------------
# These four stay network-free: a matching SQL row, or the "age unknown but
# this plan is age-banded" short-circuit, both return before _copay_fact
# ever calls out to retrieval.
def test_copay_fact_inside_a_band_is_found_from_records(fgii_store, monkeypatch):
    import src.eligibility.engine as engine

    monkeypatch.setattr(engine, "get_policy_store", lambda: fgii_store)

    fact = _copay_fact(FGII_UIN, "Hospitalisation", age=67)

    assert fact.status == "found"
    assert fact.value == 25
    assert fact.source == "policy records"
    assert "65-69" in fact.detail


def test_copay_fact_age_banded_plan_with_unknown_age_is_unknown_not_a_guess(
    fgii_store, monkeypatch
):
    """The regression case this whole change exists to prevent.

    Picking a band without knowing the customer's age would be exactly the
    kind of silent default the eligibility engine was rewritten to remove.
    """
    import src.eligibility.engine as engine

    monkeypatch.setattr(engine, "get_policy_store", lambda: fgii_store)

    fact = _copay_fact(FGII_UIN, "Hospitalisation", age=None)

    assert fact.status == "unknown"
    assert fact.value is None


def test_copay_fact_unbanded_plan_is_unaffected_by_a_missing_age(monkeypatch, tmp_path):
    """A plan with no age bands must not start demanding an age it never needed."""
    import src.eligibility.engine as engine

    store = PolicyDataStore(tmp_path / "policy.db")
    store.add_copayment(
        policy_uin="SHAHLIP22027V032122", insurer="Star Health",
        condition="all claims", copay_percent=5, age_min=None, age_max=None, page=8,
    )
    monkeypatch.setattr(engine, "get_policy_store", lambda: store)

    fact = _copay_fact("SHAHLIP22027V032122", "Cataract Surgery", age=None)

    assert fact.status == "found"
    assert fact.value == 5


def test_copay_fact_outside_every_band_is_not_a_guessed_zero(fgii_store, monkeypatch):
    """39 matches no band; this must not silently become "no co-pay"."""
    import src.eligibility.engine as engine

    monkeypatch.setattr(engine, "get_policy_store", lambda: fgii_store)

    all_rows = fgii_store.get_copayments(FGII_UIN)
    matched = fgii_store.get_copayments(FGII_UIN, age=39)

    assert len(all_rows) == 4
    assert matched == []
