"""Deterministic claim eligibility checklist.

WHY this exists: an LLM asked "is this claim eligible" will happily give a
plausible-sounding but wrong answer. This engine runs the same 8-step
checklist a human claims assessor would, using only SQL lookups, RAG
citations, and arithmetic -- the LLM is only ever handed the finished
result to explain in plain language, never asked to decide it.
"""
from __future__ import annotations

import logging

from config import settings
from src.crm import CRMStore
from src.policy_data import PolicyDataStore
from src.tools.calc_tools import (
    calculate_payable_amount,
    compute_sum_insured_balance,
    compute_waiting_period,
)
from src.eligibility import parsing
from src.tools.rag_tools import check_coverage, check_exclusion, check_waiting_period

logger = logging.getLogger(__name__)

_crm_store: CRMStore | None = None
_policy_store: PolicyDataStore | None = None


def _get_crm_store() -> CRMStore:
    global _crm_store
    if _crm_store is None:
        _crm_store = CRMStore(settings.crm_db_path)
    return _crm_store


def _get_policy_store() -> PolicyDataStore:
    global _policy_store
    if _policy_store is None:
        _policy_store = PolicyDataStore(settings.crm_db_path)
    return _policy_store


# Words like "surgery" or "treatment" appear in almost every clause in a
# policy document, so on their own they identify nothing. Matching on them
# would make "cataract surgery" false-positive against a "cosmetic surgery"
# exclusion. They're dropped before matching; only the words that actually
# name the procedure are checked, and ALL of those must be present.
GENERIC_MEDICAL_WORDS = {
    "surgery", "surgical", "treatment", "procedure", "therapy",
    "expenses", "medical", "hospitalization", "hospitalisation", "care",
    "inpatient", "outpatient", "patient", "admission", "hospital",
    # Surgical approach, not identity. A bill says "Laparoscopic
    # Appendectomy" where the policy just lists "Appendectomy"; requiring
    # the approach word too rejected a covered procedure.
    "laparoscopic", "laparoscopy", "endoscopic", "arthroscopic", "robotic",
}


def _specific_terms(treatment: str) -> list[str]:
    """Return the words in a treatment name that actually identify it."""
    normalized = treatment.lower().replace("_", " ")
    return [
        word for word in normalized.split()
        if len(word) > 3 and word not in GENERIC_MEDICAL_WORDS
    ]


def _mentions_treatment(text: str, treatment: str) -> bool:
    """Check whether a clause's text is actually about this treatment.

    Requires every specific (non-generic) word in the treatment name to
    appear in the clause -- a single generic word like "surgery" matching
    is not enough, since that matches almost any clause in the document.
    A treatment name with no specific words at all (too vague to judge)
    never matches.
    """
    words = _specific_terms(treatment)
    if not words:
        return False

    lowered_text = text.lower()
    return all(word in lowered_text for word in words)


def _find_applicable_copay(copayments: list[dict], treatment: str) -> dict | None:
    """Prefer a co-pay row naming this treatment, else fall back to a blanket one."""
    normalized = treatment.lower().replace("_", " ")
    for row in copayments:
        condition = (row.get("condition") or "").lower()
        if normalized in condition:
            return row
    for row in copayments:
        if "all claims" in (row.get("condition") or "").lower():
            return row
    return copayments[0] if copayments else None


def _check_policy_active(policy: dict) -> str | None:
    """Step 1: reject outright if the policy itself is not active."""
    if policy["status"] != "active":
        return f"Policy status is '{policy['status']}', not active."
    return None


def _check_coverage_and_exclusion(
    treatment: str,
    policy_uin: str,
) -> tuple[str | None, list[dict]]:
    """Steps 2-3: an explicit exclusion rejects the claim; nothing else does.

    These policies cover hospitalisation in general and name only what they
    exclude or cap -- there is no exhaustive list of covered procedures. The
    old rule ("reject unless a coverage clause names the treatment") therefore
    rejected ordinary claims: an appendectomy is covered but never mentioned
    by name anywhere in the wording. Sub-limit, co-pay, waiting-period and
    sum-insured checks still run after this, so a claim is not waved through.
    """
    exclusion_hits = check_exclusion.invoke({"treatment": treatment, "policy_uin": policy_uin})

    # Every exclusion clause is checked, not just the first two. An exclusion
    # is now the only thing that can reject a claim, so one ranked further
    # down the list would otherwise be missed and the claim wrongly approved.
    excluded = [
        hit for hit in exclusion_hits if _mentions_treatment(hit["text"], treatment)
    ]
    if excluded:
        return f"'{treatment}' appears in the policy's exclusion list.", excluded[:2]

    # Not excluded, so it is covered. Prefer a clause that names the treatment
    # for the citation, otherwise cite the nearest coverage clauses.
    coverage_hits = check_coverage.invoke({"treatment": treatment, "policy_uin": policy_uin})
    naming = [
        hit for hit in coverage_hits if _mentions_treatment(hit["text"], treatment)
    ]
    return None, (naming or coverage_hits)[:2]


def _waiting_period_months(policy_uin: str, treatment: str) -> int | None:
    """SQL first, then the policy text, then None (no waiting period known)."""
    row = _get_policy_store().find_waiting_period(policy_uin, treatment)
    if row is not None:
        return row["waiting_period_months"]

    # The clause must name this treatment. Without that guard a cataract bill
    # picked up the 48-month pre-existing-disease period from a neighbouring
    # clause and was wrongly rejected.
    terms = _specific_terms(treatment)
    if not terms:
        return None

    hits = check_waiting_period.invoke(
        {"condition": treatment, "policy_uin": policy_uin}
    )
    for term in terms:
        months, hit = parsing.first_match(
            hits, parsing.parse_waiting_period_months, keyword=term
        )
        if months is not None:
            logger.info(
                "waiting_period_from_rag uin=%s treatment=%r months=%s page=%s",
                policy_uin, treatment, months, (hit or {}).get("page"),
            )
            return months
    return None


def _check_waiting_period(policy: dict, treatment: str) -> str | None:
    """Step 4: only blocks the claim if a waiting period can be established."""
    months = _waiting_period_months(policy["policy_number"], treatment)
    if months is None:
        # Safe default: nothing on record in SQL or in the wording, so no
        # waiting period is applied rather than inventing one.
        return None

    tracker = compute_waiting_period(policy["start_date"], months)
    if not tracker["is_eligible"]:
        return (
            f"Waiting period for '{treatment}' is not yet complete. "
            f"Eligible from {tracker['eligible_date']}."
        )
    return None


def _lookup_sub_limit(policy_uin: str, treatment: str) -> float | None:
    """SQL, then the wording, then None meaning no cap applies."""
    row = _get_policy_store().find_sub_limit(policy_uin, treatment)
    if row is not None:
        return row["limit_amount"]

    # The clause has to name the treatment, otherwise any rupee figure on the
    # page would be mistaken for this treatment's cap.
    hits = check_coverage.invoke({"treatment": treatment, "policy_uin": policy_uin})
    for term in _specific_terms(treatment):
        amount, hit = parsing.first_match(hits, parsing.parse_rupee_amount, keyword=term)
        if amount is not None:
            logger.info(
                "sub_limit_from_rag uin=%s treatment=%r amount=%s page=%s",
                policy_uin, treatment, amount, (hit or {}).get("page"),
            )
            return amount
    return None


def _lookup_copay(policy_uin: str, treatment: str) -> float | None:
    """SQL, then the wording, then None which the calculator reads as 0%."""
    copay_row = _find_applicable_copay(
        _get_policy_store().get_copayments(policy_uin), treatment
    )
    if copay_row is not None:
        return copay_row["copay_percent"]

    hits = check_coverage.invoke(
        {"treatment": "co-payment", "policy_uin": policy_uin}
    )
    percent, hit = parsing.first_match(hits, parsing.parse_percent, keyword="co-pay")
    if percent is not None:
        logger.info(
            "copay_from_rag uin=%s percent=%s page=%s",
            policy_uin, percent, (hit or {}).get("page"),
        )
    return percent


def _lookup_deductible(policy_uin: str) -> float:
    """SQL, then the wording, then 0 -- most plans have no deductible."""
    row = _get_policy_store().find_deductible(policy_uin)
    if row is not None:
        return row["deductible_amount"]

    hits = check_coverage.invoke(
        {"treatment": "deductible", "policy_uin": policy_uin}
    )
    amount, hit = parsing.first_match(
        hits, parsing.parse_rupee_amount, keyword="deductible"
    )
    if amount is not None:
        logger.info(
            "deductible_from_rag uin=%s amount=%s page=%s",
            policy_uin, amount, (hit or {}).get("page"),
        )
        return amount
    return 0


def _get_deduction_context(
    policy_uin: str,
    treatment: str,
    policy_id: str,
    customer_id: str,
) -> tuple:
    """Fetch the deduction figures, reading SQL first and the wording after.

    Each lookup is SQL -> RAG -> safe default, so a customer whose policy has
    no curated rows still gets the limits their wording actually states.
    """
    sub_limit = _lookup_sub_limit(policy_uin, treatment)
    copay_percent = _lookup_copay(policy_uin, treatment)
    deductible = _lookup_deductible(policy_uin)

    balance = compute_sum_insured_balance(policy_id, customer_id)
    remaining_sum_insured = balance.get("remaining_balance", 0)

    return sub_limit, copay_percent, deductible, remaining_sum_insured


def check_eligibility(bill_data: dict, customer_id: str, policy_id: str) -> dict:
    """Run the full eligibility checklist for one bill against one policy."""
    policy = _get_crm_store().get_policy(policy_id)
    if policy is None or policy["customer_id"] != customer_id:
        return {"eligible": False, "rejection_reason": "Policy not found for this customer."}

    treatment = bill_data.get("treatment") or bill_data.get("diagnosis") or ""
    bill_amount = float(bill_data.get("total_amount") or 0)
    policy_uin = policy["policy_number"]

    rejection_reason = _check_policy_active(policy)
    if rejection_reason:
        return _rejected(bill_amount, rejection_reason, [])

    rejection_reason, evidence = _check_coverage_and_exclusion(treatment, policy_uin)
    if rejection_reason:
        return _rejected(bill_amount, rejection_reason, evidence)

    rejection_reason = _check_waiting_period(policy, treatment)
    if rejection_reason:
        return _rejected(bill_amount, rejection_reason, evidence)

    sub_limit, copay_percent, deductible, remaining_sum_insured = _get_deduction_context(
        policy_uin, treatment, policy_id, customer_id
    )
    if remaining_sum_insured <= 0:
        return _rejected(bill_amount, "Sum insured for this policy year is exhausted.", evidence)

    payable = calculate_payable_amount.invoke({
        "bill_amount": bill_amount,
        "sub_limit": sub_limit,
        "copay_percent": copay_percent,
        "deductible": deductible,
        "remaining_sum_insured": remaining_sum_insured,
    })
    covered_amount = bill_amount - payable["deductions"]["sub_limit_reduction"]

    logger.info(
        "eligibility_checked customer_id=%s policy_id=%s treatment=%r eligible=True payable=%s",
        customer_id, policy_id, treatment, payable["payable_amount"],
    )
    return {
        "eligible": True,
        "bill_amount": bill_amount,
        "covered_amount": round(covered_amount, 2),
        "copay_amount": payable["deductions"]["copay_amount"],
        "estimated_payable": payable["payable_amount"],
        "rejection_reason": None,
        "evidence": evidence,
    }


def _rejected(bill_amount: float, reason: str, evidence: list[dict]) -> dict:
    """Build the standard rejection response shape."""
    logger.info("eligibility_checked eligible=False reason=%r", reason)
    return {
        "eligible": False,
        "bill_amount": bill_amount,
        "covered_amount": 0,
        "copay_amount": 0,
        "estimated_payable": 0,
        "rejection_reason": reason,
        "evidence": evidence,
    }
