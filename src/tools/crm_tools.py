"""Agent-facing tools over the CRM store: customers, policies, claims.

WHY these tools take no customer_id: it is injected at invoke time via
ToolRuntime.context, so it never appears in the schema the LLM sees. The
model therefore cannot reach another customer's records, and cannot ask for
an ID the employee already entered on the claim form.
"""
from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.crm import get_crm_store


@tool
def get_customer(runtime: ToolRuntime) -> dict:
    """Return the profile of the customer whose claim is being analysed: name,
    email, phone, and date of birth."""
    result = get_crm_store().get_customer(runtime.context.customer_id)
    return result or {"error": "Customer not found."}


@tool
def get_policies(runtime: ToolRuntime) -> list[dict]:
    """Return every health insurance policy this customer holds, including
    policy_id, policy_number (the UIN), plan_name, sum_insured, start_date,
    and end_date. Call this first for any question about their cover — it
    gives you the policy_id and UIN the other tools need."""
    return get_crm_store().get_policies(runtime.context.customer_id)


@tool
def get_claims(runtime: ToolRuntime, policy_id: str | None = None) -> list[dict]:
    """Return this customer's claim history: claim_id, diagnosis,
    claim_amount, eligible_amount, status (pending/approved/rejected), and
    claim_date. Pass policy_id to filter to a single policy."""
    return get_crm_store().get_claims(runtime.context.customer_id, policy_id)
