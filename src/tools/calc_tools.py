"""Agent-facing calculation tools — pure Python, no LLM.

The LLM never does arithmetic on money or dates in this project. It calls
these functions and reports their results, so a customer sees the same
number every time they ask the same question.
"""
from __future__ import annotations

from calendar import monthrange
from datetime import date, datetime

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from config import settings
from src.crm import get_crm_store
from src.policy_data import get_policy_store

def _parse_date(value: str | date) -> date:
    """Accept either a date/datetime object or an ISO-formatted string."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value[:10])


def _add_months(start: date, months: int) -> date:
    """Add whole months to a date, clamping to the shorter month's length."""
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, monthrange(year, month)[1])
    return date(year, month, day)


# ---------------------------------------------------------------------------
# 1. Waiting period date math
# ---------------------------------------------------------------------------
def compute_waiting_period(
    policy_start_date: str | date,
    waiting_period_months: int,
    today: str | date | None = None,
) -> dict:
    """Core date math, callable directly by the eligibility engine."""
    start = _parse_date(policy_start_date)
    as_of = _parse_date(today) if today else date.today()

    eligible_date = _add_months(start, waiting_period_months)
    days_remaining = max(0, (eligible_date - as_of).days)

    return {
        "eligible_date": eligible_date.isoformat(),
        "days_remaining": days_remaining,
        "is_eligible": as_of >= eligible_date,
    }


@tool
def waiting_period_tracker(
    policy_id: str,
    condition: str,
    runtime: ToolRuntime,
    waiting_period_months: int | None = None,
) -> dict:
    """Give the exact date cover for a condition begins on the signed-in
    customer's policy, returning eligible_date, days_remaining and is_eligible.
    Pass policy_id from get_policies and the condition (e.g. maternity,
    cataract). The waiting period is looked up automatically — only pass
    waiting_period_months if a clause stated a number this policy's records
    lack. Always finish a waiting-period question with this tool."""
    policy = get_crm_store().get_policy(policy_id)
    if policy is None or policy["customer_id"] != runtime.context.customer_id:
        return {"error": "Policy not found for this customer."}

    # Most waiting periods live in a table that was routed to SQL, not in the
    # clause text, so the number usually has to come from here.
    if waiting_period_months is None:
        record = get_policy_store().find_waiting_period(
            policy["policy_number"], condition
        )
        if record is None:
            return {
                "error": (
                    f"No waiting period on record for '{condition}' under this "
                    f"policy."
                )
            }
        waiting_period_months = record["waiting_period_months"]

    result = compute_waiting_period(policy["start_date"], waiting_period_months)
    result["condition"] = condition
    result["waiting_period_months"] = waiting_period_months
    return result


# ---------------------------------------------------------------------------
# 2. Sum insured balance
# ---------------------------------------------------------------------------
def compute_sum_insured_balance(policy_id: str, customer_id: str) -> dict:
    """Core sum-insured arithmetic, callable directly by the eligibility engine.

    WHY separate from the tool below: the engine already knows the customer_id
    and runs outside any agent turn, so it has no ToolRuntime to read from.
    """
    policy = get_crm_store().get_policy(policy_id)
    if policy is None or policy["customer_id"] != customer_id:
        return {"error": "Policy not found for this customer."}

    start = _parse_date(policy["start_date"])
    end = _parse_date(policy["end_date"])

    claims = get_crm_store().get_claims(customer_id, policy_id)
    claims_used = sum(
        claim["eligible_amount"] or claim["claim_amount"]
        for claim in claims
        if claim["status"] == "approved"
        and start <= _parse_date(claim["claim_date"]) <= end
    )

    # The deductible and co-pay are policy facts held in SQL. They are returned
    # here because calculate_payable_amount cannot look them up itself, and
    # without them the agent silently treats both as zero -- which told a
    # top-up customer their whole bill was payable when the deductible
    # actually left them nothing.
    policy_uin = policy["policy_number"]
    deductible_row = get_policy_store().find_deductible(policy_uin)
    copayments = get_policy_store().get_copayments(policy_uin)

    sum_insured = policy["sum_insured"]
    return {
        "sum_insured": sum_insured,
        "claims_used": claims_used,
        "remaining_balance": max(0, sum_insured - claims_used),
        "deductible": deductible_row["deductible_amount"] if deductible_row else 0,
        "copay_percent": copayments[0]["copay_percent"] if copayments else 0,
    }


@tool
def sum_insured_balance(policy_id: str, runtime: ToolRuntime) -> dict:
    """Report what the signed-in customer's policy still pays: sum_insured,
    claims_used, remaining_balance, plus the policy's deductible and
    copay_percent. Call this before calculate_payable_amount and pass its
    deductible and copay_percent straight through — they are often non-zero.
    Get policy_id from get_policies first."""
    return compute_sum_insured_balance(policy_id, runtime.context.customer_id)


# ---------------------------------------------------------------------------
# 5. Payable amount
# ---------------------------------------------------------------------------
def _apply_deductions(
    covered_amount: float,
    copay_percent: float,
    deductible: float,
    remaining_sum_insured: float,
) -> dict:
    """Apply copay and deductible in order, then cap to remaining sum insured."""
    copay_amount = round(covered_amount * (copay_percent / 100), 2)
    after_copay = covered_amount - copay_amount
    after_deductible = max(0, after_copay - deductible)
    return {
        "copay_amount": copay_amount,
        "after_copay": after_copay,
        "after_deductible": after_deductible,
        "payable_amount": min(after_deductible, remaining_sum_insured),
    }


@tool
def calculate_payable_amount(
    bill_amount: float,
    remaining_sum_insured: float,
    sub_limit: float | None = None,
    copay_percent: float | None = None,
    deductible: float | None = None,
) -> dict:
    """Compute the actual amount the insurer pays after applying sub-limit,
    co-payment, and deductible in order. Returns payable_amount and a breakdown
    of deductions. Never calculate this yourself — always call this tool. Take
    remaining_sum_insured, deductible and copay_percent from
    sum_insured_balance; omitting them understates what the customer owes."""
    sub_limit = bill_amount if sub_limit is None else sub_limit
    copay_percent = copay_percent or 0
    deductible = deductible or 0

    covered_amount = min(bill_amount, sub_limit)
    sub_limit_reduction = bill_amount - covered_amount

    d = _apply_deductions(covered_amount, copay_percent, deductible, remaining_sum_insured)
    return {
        "payable_amount": round(d["payable_amount"], 2),
        "deductions": {
            "sub_limit_reduction": round(sub_limit_reduction, 2),
            "copay_amount": d["copay_amount"],
            "deductible_amount": round(min(deductible, d["after_copay"]), 2),
            "capped_by_remaining_sum_insured": d["after_deductible"] > remaining_sum_insured,
        },
    }
