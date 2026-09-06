"""Deterministic claim eligibility checks and payable-amount calculation."""
from __future__ import annotations

import logging
import re

from . import parsing
from .facts import (
    FOUND,
    SOURCE_SQL,
    SOURCE_WORDING,
    PolicyFact,
    found,
    not_applicable,
    unknown,
)
from src.crm import get_crm_store
from src.policy_data import get_policy_store
from src.tools.calc_tools import (
    calculate_payable_amount,
    compute_age,
    compute_sum_insured_balance,
    compute_waiting_period,
)
from src.tools.rag_tools import check_coverage, check_exclusion, check_waiting_period

logger = logging.getLogger(__name__)

ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
NEEDS_MORE_INFO = "needs_more_info"

GENERIC_MEDICAL_WORDS = {
    "surgery", "surgical", "treatment", "procedure", "therapy",
    "expenses", "medical", "hospitalization", "hospitalisation", "care",
    "inpatient", "outpatient", "patient", "admission", "hospital",
    "laparoscopic", "laparoscopy", "endoscopic", "arthroscopic", "robotic",
}


def _specific_terms(treatment: str) -> list[str]:
    """Return treatment terms that identify the procedure or condition."""
    normalized = treatment.lower().replace("_", " ")
    return [
        word for word in normalized.split()
        if len(word) > 3 and word not in GENERIC_MEDICAL_WORDS
    ]


def _mentions_treatment(text: str, treatment: str) -> bool:
    """Check whether a clause contains all specific treatment terms."""
    words = _specific_terms(treatment)
    if not words:
        return False
    lowered_text = text.lower()
    return all(word in lowered_text for word in words)


def _check_policy_in_force(policy: dict, treatment_date: str | None) -> str | None:
    """Return a policy-period error, or None when the policy is in force."""
    if policy["status"] != "active":
        return f"Policy status is '{policy['status']}', not active."

    if not treatment_date:
        return None

    date_only = str(treatment_date)[:10]
    start_date = str(policy["start_date"])[:10]
    end_date = str(policy["end_date"])[:10]
    if date_only < start_date:
        return f"Treatment date {date_only} is before the policy start date {start_date}."
    if date_only > end_date:
        return f"Treatment date {date_only} is after the policy end date {end_date}."
    return None


def _check_exclusion(treatment: str, policy_uin: str) -> tuple[bool, list[dict]]:
    """Return whether retrieved exclusion clauses name the treatment."""
    hits = check_exclusion.invoke({"treatment": treatment, "policy_uin": policy_uin})
    excluded = [hit for hit in hits if _mentions_treatment(hit["text"], treatment)]
    return bool(excluded), excluded[:2]


BASE_COVER_PATTERNS = (
    re.compile(r"hospitali[sz]ation expenses", re.I),
    re.compile(r"in-?patient (care|treatment|hospitali)", re.I),
    re.compile(r"medical expenses.{0,80}hospitali", re.I | re.S),
    re.compile(r"shall (indemnify|pay|reimburse).{0,120}hospitali", re.I | re.S),
    re.compile(r"expenses.{0,60}incurred.{0,60}hospitali", re.I | re.S),
)


def _grants_base_cover(text: str) -> bool:
    """Return True when a clause grants general hospitalisation cover."""
    return any(pattern.search(text or "") for pattern in BASE_COVER_PATTERNS)


def _check_coverage(treatment: str, policy_uin: str) -> PolicyFact:
    """Resolve coverage from curated facts, wording, or base cover."""
    store = get_policy_store()
    sql_row = store.find_sub_limit(policy_uin, treatment)
    if sql_row is not None:
        return found(
            "coverage",
            True,
            SOURCE_SQL,
            f"'{treatment}' has a sub-limit on record, so the plan covers it.",
        )

    hits = check_coverage.invoke({"treatment": treatment, "policy_uin": policy_uin})
    naming = [hit for hit in hits if _mentions_treatment(hit["text"], treatment)]
    if naming:
        return found(
            "coverage",
            True,
            SOURCE_WORDING,
            f"The wording names '{treatment}' as covered.",
            evidence=naming[:2],
        )

    granting = [hit for hit in hits if _grants_base_cover(hit["text"])]
    if granting:
        return found(
            "coverage",
            True,
            SOURCE_WORDING,
            f"'{treatment}' falls under the plan's hospitalisation cover; confirm the admission was in-patient.",
            evidence=granting[:2],
        )

    return unknown(
        "coverage",
        f"No clause granting cover could be retrieved for '{treatment}'.",
    )


def _waiting_period_fact(policy_uin: str, treatment: str) -> PolicyFact:
    """Resolve waiting period from SQL, policy wording, or unknown."""
    store = get_policy_store()
    row = store.find_waiting_period(policy_uin, treatment)
    if row is not None:
        return found(
            "waiting_period",
            row["waiting_period_months"],
            SOURCE_SQL,
            f"{row['waiting_period_months']}-month waiting period on record for '{row.get('condition') or treatment}'.",
        )

    on_record = store.get_waiting_periods(policy_uin)
    if on_record:
        listed = ", ".join(r["condition"] for r in on_record if r.get("condition"))
        return not_applicable(
            "waiting_period",
            f"The plan's waiting-period schedule does not list '{treatment}' among {listed}.",
        )

    terms = _specific_terms(treatment)
    hits = check_waiting_period.invoke({"condition": treatment, "policy_uin": policy_uin})
    for term in terms:
        months, hit = parsing.first_match(
            hits, parsing.parse_waiting_period_months, keyword=term
        )
        if months is not None:
            logger.info(
                "waiting_period_from_rag uin=%s treatment=%r months=%s page=%s",
                policy_uin, treatment, months, (hit or {}).get("page"),
            )
            return found(
                "waiting_period",
                months,
                SOURCE_WORDING,
                f"The wording states a {months}-month waiting period for '{treatment}'.",
                evidence=[hit] if hit else [],
            )

    if hits and terms:
        return not_applicable(
            "waiting_period",
            f"The waiting-period clauses do not name '{treatment}'.",
        )

    return unknown(
        "waiting_period",
        f"No waiting-period clause could be read for '{treatment}'.",
    )


def _waiting_period_breach(policy: dict, fact: PolicyFact) -> str | None:
    """Return the waiting-period breach, or None."""
    if fact.status != FOUND:
        return None
    tracker = compute_waiting_period(policy["start_date"], fact.value)
    if tracker["is_eligible"]:
        return None
    return (
        f"Waiting period of {fact.value} months is not yet complete. "
        f"Cover begins {tracker['eligible_date']}."
    )


def _sub_limit_fact(policy_uin: str, treatment: str) -> PolicyFact:
    """Resolve the sub-limit from SQL, policy wording, or unknown."""
    row = get_policy_store().find_sub_limit(policy_uin, treatment)
    if row is not None:
        return found(
            "sub_limit",
            row["limit_amount"],
            SOURCE_SQL,
            f"Sub-limit of Rs {row['limit_amount']:,.0f} on record for '{row.get('treatment') or treatment}'.",
        )

    hits = check_coverage.invoke({"treatment": treatment, "policy_uin": policy_uin})
    for term in _specific_terms(treatment):
        amount, hit = parsing.first_match(hits, parsing.parse_rupee_amount, keyword=term)
        if amount is not None:
            logger.info(
                "sub_limit_from_rag uin=%s treatment=%r amount=%s page=%s",
                policy_uin, treatment, amount, (hit or {}).get("page"),
            )
            return found(
                "sub_limit",
                amount,
                SOURCE_WORDING,
                f"The wording caps '{treatment}' at Rs {amount:,.0f}.",
                evidence=[hit] if hit else [],
            )

    if hits:
        return not_applicable(
            "sub_limit",
            f"No clause caps '{treatment}'.",
        )

    return unknown(
        "sub_limit",
        f"No coverage wording could be retrieved, so any cap on '{treatment}' is unestablished.",
    )


def _find_applicable_copay(copayments: list[dict], treatment: str) -> dict | None:
    """Prefer a treatment-specific co-pay, then a blanket row."""
    normalized = treatment.lower().replace("_", " ")
    for row in copayments:
        condition = (row.get("condition") or "").lower()
        if normalized in condition:
            return row
    for row in copayments:
        if "all claims" in (row.get("condition") or "").lower():
            return row
    return copayments[0] if copayments else None


def _age_band_text(row: dict) -> str:
    """Render a co-pay age band."""
    age_min, age_max = row.get("age_min"), row.get("age_max")
    if age_min is not None and age_max is not None:
        return f"age {age_min}-{age_max}"
    if age_min is not None:
        return f"age {age_min}+"
    if age_max is not None:
        return f"age up to {age_max}"
    return ""


def _copay_fact(policy_uin: str, treatment: str, age: int | None) -> PolicyFact:
    """Resolve the co-pay from age-filtered SQL rows or policy wording."""
    store = get_policy_store()
    all_rows = store.get_copayments(policy_uin)
    is_age_banded = any(
        row.get("age_min") is not None or row.get("age_max") is not None
        for row in all_rows
    )
    if is_age_banded and age is None:
        return unknown(
            "copay",
            "This plan's co-payment depends on the customer's age, which could not be determined.",
        )

    row = _find_applicable_copay(store.get_copayments(policy_uin, age=age), treatment)
    if row is not None:
        band = _age_band_text(row)
        return found(
            "copay",
            row["copay_percent"],
            SOURCE_SQL,
            f"Co-payment of {row['copay_percent']}% on record for '{row.get('condition') or 'all claims'}'"
            + (f" ({band})" if band else "") + ".",
        )

    hits = check_coverage.invoke({"treatment": "co-payment", "policy_uin": policy_uin})
    percent, hit = parsing.first_match(hits, parsing.parse_percent, keyword="co-pay")
    if percent is not None:
        logger.info(
            "copay_from_rag uin=%s percent=%s page=%s",
            policy_uin, percent, (hit or {}).get("page"),
        )
        return found(
            "copay",
            percent,
            SOURCE_WORDING,
            f"The wording states a {percent}% co-payment.",
            evidence=[hit] if hit else [],
        )

    if hits:
        return not_applicable("copay", "No co-payment clause applies to this plan.")

    return unknown(
        "copay",
        "No co-payment wording could be retrieved.",
    )


def _deductible_fact(policy_uin: str) -> PolicyFact:
    """Resolve the deductible from SQL, policy wording, or unknown."""
    row = get_policy_store().find_deductible(policy_uin)
    if row is not None:
        return found(
            "deductible",
            row["deductible_amount"],
            SOURCE_SQL,
            f"Deductible of Rs {row['deductible_amount']:,.0f} on record.",
        )

    hits = check_coverage.invoke({"treatment": "deductible", "policy_uin": policy_uin})
    amount, hit = parsing.first_match(
        hits, parsing.parse_rupee_amount, keyword="deductible"
    )
    if amount is not None:
        logger.info(
            "deductible_from_rag uin=%s amount=%s page=%s",
            policy_uin, amount, (hit or {}).get("page"),
        )
        return found(
            "deductible",
            amount,
            SOURCE_WORDING,
            f"The wording states a Rs {amount:,.0f} deductible.",
            evidence=[hit] if hit else [],
        )

    if hits:
        return not_applicable(
            "deductible", "This plan carries no deductible; it pays from rupee one."
        )

    return unknown(
        "deductible",
        "No wording could be retrieved, so the deductible is unestablished.",
    )


def _result(
    status: str,
    bill_amount: float,
    reason: str,
    evidence: list[dict],
    facts: list[PolicyFact],
    missing: list[str] | None = None,
    covered_amount: float = 0,
    copay_amount: float = 0,
    payable: float | None = 0,
    deductions: dict | None = None,
) -> dict:
    """Build the standard eligibility result."""
    logger.info("eligibility_checked status=%s reason=%r", status, reason)
    return {
        "status": status,
        "bill_amount": bill_amount,
        "covered_amount": round(covered_amount, 2),
        "copay_amount": copay_amount,
        "estimated_payable": payable,
        "reason": reason,
        "evidence": evidence,
        "missing_information": missing or [],
        "facts": {fact.name: fact.to_dict() for fact in facts},
        "deductions": deductions or {},
    }


def _needs_more_info(
    bill_amount: float,
    facts: list[PolicyFact],
    evidence: list[dict],
    reason: str,
    missing: list[str],
) -> dict:
    """Return a result without a payable figure when required evidence is missing."""
    return _result(
        NEEDS_MORE_INFO,
        bill_amount,
        reason,
        evidence,
        facts,
        missing=missing,
        payable=None,
    )


def _customer_age_at_treatment(customer_id: str, treatment_date: str | None) -> int | None:
    """Return customer age on the treatment date."""
    customer = get_crm_store().get_customer(customer_id)
    date_of_birth = customer.get("date_of_birth") if customer else None
    if not date_of_birth or not treatment_date:
        return None
    return compute_age(date_of_birth, treatment_date)


def _policy_gate(
    policy: dict | None,
    customer_id: str,
    treatment: str,
    bill_amount: float,
    bill_data: dict,
) -> dict | None:
    """Validate policy ownership, policy period, and claim treatment."""
    if policy is None or policy["customer_id"] != customer_id:
        return _result(
            INELIGIBLE, bill_amount,
            "This policy is not held by this customer.", [], [],
        )

    not_in_force = _check_policy_in_force(policy, bill_data.get("admission_date"))
    if not_in_force:
        return _result(INELIGIBLE, bill_amount, not_in_force, [], [])

    if not _specific_terms(treatment):
        return _needs_more_info(
            bill_amount, [], [],
            f"The claim gives '{treatment or 'no treatment'}', which does not name a specific procedure or condition to assess.",
            ["A specific treatment or diagnosis on the claim form."],
        )
    return None


def _clause_checks(
    policy: dict, treatment: str, bill_amount: float
) -> tuple[dict | None, PolicyFact | None, PolicyFact | None, list[dict]]:
    """Run exclusion, coverage, and waiting-period checks."""
    policy_uin = policy["policy_number"]
    is_excluded, exclusion_evidence = _check_exclusion(treatment, policy_uin)
    if is_excluded:
        return _result(
            INELIGIBLE, bill_amount,
            f"'{treatment}' appears in this policy's exclusion list.",
            exclusion_evidence, [],
        ), None, None, []

    coverage = _check_coverage(treatment, policy_uin)
    if not coverage.is_known:
        return _needs_more_info(
            bill_amount, [coverage], coverage.evidence,
            f"Cover for '{treatment}' could not be confirmed from this policy's wording or records.",
            [coverage.detail],
        ), None, None, []
    evidence = list(coverage.evidence)

    waiting = _waiting_period_fact(policy_uin, treatment)
    if not waiting.is_known:
        return _needs_more_info(
            bill_amount, [coverage, waiting], evidence,
            f"The waiting period for '{treatment}' could not be established.",
            [waiting.detail],
        ), None, None, []
    evidence.extend(waiting.evidence)

    breach = _waiting_period_breach(policy, waiting)
    if breach:
        return _result(
            INELIGIBLE, bill_amount, breach, evidence, [coverage, waiting]
        ), None, None, []

    return None, coverage, waiting, evidence


def _resolve_deductions(
    policy_uin: str,
    treatment: str,
    bill_amount: float,
    base: list[PolicyFact],
    evidence: list[dict],
    age: int | None,
) -> tuple[dict | None, list[PolicyFact]]:
    """Resolve sub-limit, co-pay, and deductible facts."""
    sub_limit = _sub_limit_fact(policy_uin, treatment)
    copay = _copay_fact(policy_uin, treatment, age)
    deductible = _deductible_fact(policy_uin)
    facts = [*base, sub_limit, copay, deductible]

    unresolved = [fact for fact in facts if not fact.is_known]
    if unresolved:
        return _needs_more_info(
            bill_amount,
            facts,
            evidence,
            "The payable amount cannot be calculated until "
            + ", ".join(fact.name.replace("_", " ") for fact in unresolved)
            + " is established.",
            [fact.detail for fact in unresolved],
        ), facts

    for fact in (sub_limit, copay, deductible):
        evidence.extend(fact.evidence)
    return None, facts


def _price_the_claim(
    policy_id: str,
    customer_id: str,
    treatment: str,
    bill_amount: float,
    facts: list[PolicyFact],
    evidence: list[dict],
) -> dict:
    """Resolve remaining cover and calculate the payable amount."""
    sub_limit, copay, deductible = facts[-3], facts[-2], facts[-1]
    balance = compute_sum_insured_balance(
        policy_id,
        customer_id,
        treatment=treatment,
    )
    if "error" in balance:
        return _needs_more_info(
            bill_amount,
            facts,
            evidence,
            balance["error"],
            [balance["error"]],
        )

    remaining = balance.get("remaining_balance", 0)
    facts = [
        *facts,
        found(
            "remaining_sum_insured",
            remaining,
            SOURCE_SQL,
            f"Rs {remaining:,.0f} of Rs {balance.get('sum_insured', 0):,.0f} remains for this policy year.",
        ),
    ]
    if remaining <= 0:
        return _result(
            INELIGIBLE,
            bill_amount,
            "The sum insured for this policy year is exhausted.",
            evidence,
            facts,
        )

    payable = calculate_payable_amount.invoke({
        "bill_amount": bill_amount,
        "sub_limit": sub_limit.value_or(None),
        "copay_percent": copay.value_or(0),
        "deductible": deductible.value_or(0),
        "remaining_sum_insured": remaining,
    })

    return _result(
        ELIGIBLE,
        bill_amount,
        f"'{treatment}' is covered, no exclusion applies, and every condition was checked.",
        evidence,
        facts,
        covered_amount=bill_amount - payable["deductions"]["sub_limit_reduction"],
        copay_amount=payable["deductions"]["copay_amount"],
        payable=payable["payable_amount"],
        deductions=payable["deductions"],
    )


def check_eligibility(bill_data: dict, customer_id: str, policy_id: str) -> dict:
    """Run the complete deterministic eligibility checklist."""
    treatment = (bill_data.get("treatment") or bill_data.get("diagnosis") or "").strip()
    bill_amount = float(bill_data.get("total_amount") or 0)
    policy = get_crm_store().get_policy(policy_id)

    blocked = _policy_gate(policy, customer_id, treatment, bill_amount, bill_data)
    if blocked:
        return blocked

    settled, coverage, waiting, evidence = _clause_checks(
        policy, treatment, bill_amount
    )
    if settled:
        return settled

    age = _customer_age_at_treatment(customer_id, bill_data.get("admission_date"))
    unresolved, facts = _resolve_deductions(
        policy["policy_number"],
        treatment,
        bill_amount,
        [coverage, waiting],
        evidence,
        age,
    )
    if unresolved:
        return unresolved

    return _price_the_claim(
        policy_id,
        customer_id,
        treatment,
        bill_amount,
        facts,
        evidence,
    )
