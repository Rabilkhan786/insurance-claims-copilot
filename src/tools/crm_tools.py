"""LangChain tools for customer, policy, and claim records."""
from __future__ import annotations

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.crm import get_crm_store


@tool
def get_customer(runtime: ToolRuntime) -> dict:
    """Return the current customer's profile."""
    result = get_crm_store().get_customer(runtime.context.customer_id)
    return result or {"error": "Customer not found."}


@tool
def get_policies(runtime: ToolRuntime) -> list[dict]:
    """Return all policies held by the current customer."""
    return get_crm_store().get_policies(runtime.context.customer_id)


@tool
def get_claims(
    runtime: ToolRuntime,
    policy_id: str | None = None,
) -> list[dict]:
    """Return the current customer's claim history."""
    return get_crm_store().get_claims(
        runtime.context.customer_id,
        policy_id,
    )
