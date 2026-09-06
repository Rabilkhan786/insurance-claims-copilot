"""Calculation tools for dates and claim amounts."""

from calendar import monthrange
from datetime import date, datetime

from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from src.crm import get_crm_store
from src.policy_data import get_policy_store


def _parse_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value[:10])


def _add_months(start: date, months: int) -> date:
    month_index = start.month - 1 + months
    year = start.year + month_index // 12
    month = month_index % 12 + 1
    day = min(start.day, monthrange(year, month)[1])
    return date(year, month, day)


def compute_age(date_of_birth: str | date, as_of: str | date) -> int:
    birth = _parse_date(date_of_birth)
    reference = _parse_date(as_of)
    age = reference.year - birth.year
    if (reference.month, reference.day) < (birth.month, birth.day):
        age -= 1
    return age


def compute_waiting_period(
    policy_start_date: str | date,
    waiting_period_months: int,
    today: str | date | None = None,
) -> dict:
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
    """Return when coverage starts for a waiting-period condition."""
    policy = get_crm_store().get_policy(policy_id)
    if policy is None or policy["customer_id"] != runtime.context.customer_id:
        return {"error": "Policy not found for this customer."}

    if waiting_period_months is None:
        record = get_policy_store().find_waiting_period(
            policy["policy_number"], condition
        )
        if record is None:
            return {"error": f"No waiting period found for '{condition}'."}
        waiting_period_months = record["waiting_period_months"]

    result = compute_waiting_period(policy["start_date"], waiting_period_months)
    result["condition"] = condition
    result["waiting_period_months"] = waiting_period_months
    return result


def compute_sum_insured_balance(policy_id: str, customer_id: str) -> dict:
    """Calculate the remaining sum insured for a customer policy."""
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
    """Return the remaining sum insured and related deductions."""
    return compute_sum_insured_balance(policy_id, runtime.context.customer_id)


def _apply_deductions(
    covered_amount: float,
    copay_percent: float,
    deductible: float,
    remaining_sum_insured: float,
) -> dict:
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
    """Calculate the insurer's payable amount and deductions."""
    sub_limit = bill_amount if sub_limit is None else sub_limit
    copay_percent = copay_percent or 0
    deductible = deductible or 0

    covered_amount = min(bill_amount, sub_limit)
    deductions = _apply_deductions(
        covered_amount,
        copay_percent,
        deductible,
        remaining_sum_insured,
    )

    return {
        "payable_amount": round(deductions["payable_amount"], 2),
        "deductions": {
            "sub_limit_reduction": round(bill_amount - covered_amount, 2),
            "copay_amount": deductions["copay_amount"],
            "deductible_amount": round(
                min(deductible, deductions["after_copay"]), 2
            ),
            "capped_by_remaining_sum_insured": (
                deductions["after_deductible"] > remaining_sum_insured
            ),
        },
    }
