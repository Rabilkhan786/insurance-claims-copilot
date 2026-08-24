"""Agent-facing tools over the CRM store: customers, policies, claims.

WHY these tools take no customer_id: it is injected from the authenticated
session via ToolRuntime.context, so it never appears in the schema the LLM
sees. The model therefore cannot pass another customer's ID, and cannot ask
the user for an ID it already has.
"""
from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from config import settings
from src.crm import CRMStore

_store: CRMStore | None = None


def _get_store() -> CRMStore:
    """Return the shared CRMStore, building it once on first use."""
    global _store
    if _store is None:
        _store = CRMStore(settings.crm_db_path)
    return _store


@tool
def get_customer(runtime: ToolRuntime) -> dict:
    """Return the signed-in customer's profile: name, email, phone, and date of
    birth. Call this to confirm who you are speaking to."""
    result = _get_store().get_customer(runtime.context.customer_id)
    return result or {"error": "Customer not found."}


@tool
def get_policies(runtime: ToolRuntime) -> list[dict]:
    """Return every health insurance policy the signed-in customer holds,
    including policy_id, policy_number (the UIN), plan_name, sum_insured,
    start_date, and end_date. Call this first for any question about "my
    policy" — it gives you the policy_id and UIN the other tools need."""
    return _get_store().get_policies(runtime.context.customer_id)


@tool
def get_claims(runtime: ToolRuntime, policy_id: str | None = None) -> list[dict]:
    """Return the signed-in customer's claim history: claim_id, diagnosis,
    claim_amount, eligible_amount, status (pending/approved/rejected), and
    claim_date. Pass policy_id to filter to a single policy."""
    return _get_store().get_claims(runtime.context.customer_id, policy_id)
