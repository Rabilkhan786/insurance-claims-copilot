"""Tests for SQLite schema creation, seeding, and CRM queries."""
import sys
from pathlib import Path

import pytest

SEED_DIR = Path(__file__).resolve().parents[1] / "Data"
sys.path.insert(0, str(SEED_DIR))

from src.crm import CRMStore
from src.decisions import DecisionStore
from src.policy_data import PolicyDataStore


def _table_names(db_path) -> set[str]:
    """Return all SQLite table names in a database file."""
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    finally:
        connection.close()
    return {row[0] for row in rows}


def test_all_tables_can_be_created_without_errors(tmp_path):
    """All stores should initialize their tables in the shared database."""
    crm_db = tmp_path / "crm.db"

    CRMStore(crm_db)
    DecisionStore(crm_db)
    PolicyDataStore(crm_db)

    expected = {
        "customers",
        "policies",
        "claims",
        "claim_decisions",
        "policy_sub_limits",
        "policy_waiting_periods",
        "policy_copayments",
        "policy_deductibles",
    }

    assert expected.issubset(_table_names(crm_db))


def test_seed_script_runs_without_errors(tmp_path, monkeypatch):
    """The demo seed script should build a fresh SQLite database."""
    from types import SimpleNamespace

    import seed

    monkeypatch.setattr(
        seed,
        "settings",
        SimpleNamespace(crm_db_path=tmp_path / "crm.db"),
    )

    seed.main()

    assert (tmp_path / "crm.db").exists()


@pytest.fixture
def crm_store(tmp_path):
    store = CRMStore(tmp_path / "crm.db")
    store.add_customer("A1", "Alice")
    store.add_customer("B1", "Bob")
    store.add_policy(
        "POL-A1",
        "A1",
        "PN-A1",
        "individual",
        "Alice's Policy",
        300000,
        "2025-01-01",
        "2026-01-01",
    )
    store.add_policy(
        "POL-B1",
        "B1",
        "PN-B1",
        "individual",
        "Bob's Policy",
        500000,
        "2025-01-01",
        "2026-01-01",
    )
    store.add_claim(
        "CLM-A1",
        "A1",
        "POL-A1",
        "hospitalization",
        20000,
        "2025-06-01",
        status="approved",
    )
    store.add_claim(
        "CLM-A2",
        "A1",
        "POL-A1",
        "hospitalization",
        15000,
        "2025-07-01",
        status="rejected",
    )
    return store


def test_get_customer_returns_correct_customer(crm_store):
    customer = crm_store.get_customer("A1")

    assert customer["name"] == "Alice"


def test_get_policies_returns_only_that_customers_policies(crm_store):
    policies = crm_store.get_policies("A1")

    assert [policy["policy_id"] for policy in policies] == ["POL-A1"]


def test_get_claims_filtered_by_status_returns_only_approved(crm_store):
    claims = crm_store.get_claims("A1")
    approved = [claim for claim in claims if claim["status"] == "approved"]

    assert [claim["claim_id"] for claim in approved] == ["CLM-A1"]
