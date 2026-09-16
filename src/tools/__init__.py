"""Agent tool registry."""
from src.tools.calc_tools import (
    calculate_payable_amount,
    sum_insured_balance,
    waiting_period_tracker,
)
from src.tools.crm_tools import get_claims, get_customer, get_policies
from src.tools.rag_tools import (
    check_coverage,
    check_exclusion,
    check_waiting_period,
)

ALL_TOOLS = [
    get_customer,
    get_policies,
    get_claims,
    check_coverage,
    check_exclusion,
    check_waiting_period,
    waiting_period_tracker,
    sum_insured_balance,
    calculate_payable_amount,
]

__all__ = ["ALL_TOOLS"]
