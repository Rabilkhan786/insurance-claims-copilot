"""Tool subpackage — every agent-facing function is decorated with @tool.

ALL_TOOLS is the single registry handed to create_agent: exactly 9 tools.
A smaller roster means the model picks correctly more often, so anything the
agent did not genuinely need has been removed rather than kept "just in case".

Customer-scoped tools read customer_id from ToolRuntime.context rather than
taking it as an argument, so scoping is enforced by the schema itself.
"""
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
    # CRM (3)
    get_customer,
    get_policies,
    get_claims,
    # RAG (3)
    check_coverage,
    check_exclusion,
    check_waiting_period,
    # Calculations (3)
    waiting_period_tracker,
    sum_insured_balance,
    calculate_payable_amount,
]

__all__ = ["ALL_TOOLS"]
