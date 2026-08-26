"""Deterministic claim eligibility checklist.

WHY this exists: an LLM asked "is this claim eligible" will happily give a
plausible-sounding but wrong answer. This engine runs the same checklist a
human claims assessor would, using only SQL lookups, RAG citations and
arithmetic -- the LLM is only ever handed the finished result to explain,
never asked to decide it.

The checklist, in order:

     1. customer owns this policy       6. resolve sub-limit
     2. policy is active and in force   7. resolve co-pay
     3. exclusion                       8. resolve deductible
     4. coverage evidence               9. resolve remaining sum insured
     5. waiting period                 10. calculate payable
                                       11. return a structured result

It answers with one of three states, never a bare yes/no:

    eligible         covered, and every condition that applies was checked
    ineligible       an exclusion, a waiting period or an inactive policy
    needs_more_info  a fact the decision depends on could not be established

That third state is the point. "No exclusion clause matched" is not evidence
that a treatment is covered, and an empty co-pay lookup is not evidence that
the co-pay is zero. Anything the engine could not establish is named in
missing_information and handed to the employee to chase, rather than being
filled in with a default that happens to favour one answer.
"""
from __future__ import annotations

import logging
import re

from src.crm import get_crm_store
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
from src.policy_data import get_policy_store
from src.tools.calc_tools import (
    calculate_payable_amount,
    compute_age,
    compute_sum_insured_balance,
    compute_waiting_period,
)
from src.tools.rag_tools import check_coverage, check_exclusion, check_waiting_period

logger = logging.getLogger(__name__)

# The three answers the engine can give.
ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
NEEDS_MORE_INFO = "needs_more_info"

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


# ---------------------------------------------------------------------------
# Step 1-2: the policy itself
# ---------------------------------------------------------------------------
def _check_policy_in_force(policy: dict, treatment_date: str | None) -> str | None:
    """Return why the policy cannot pay, or None if it is in force.

    The treatment date matters as much as the status flag: a policy that is
    active today did not necessarily cover a treatment given last year, and
    paying on an out-of-period date is one of the mistakes this checklist
    exists to catch.
    """
    if policy["status"] != "active":
        return f"Policy status is '{policy['status']}', not active."

    if not treatment_date:
        return None

    date_only = str(treatment_date)[:10]
    if date_only < str(policy["start_date"])[:10]:
        return (
            f"Treatment date {date_only} is before the policy start date "
            f"{str(policy['start_date'])[:10]}."
        )
    if date_only > str(policy["end_date"])[:10]:
        return (
            f"Treatment date {date_only} is after the policy end date "
            f"{str(policy['end_date'])[:10]}."
        )
    return None


# ---------------------------------------------------------------------------
# Step 3: exclusion
# ---------------------------------------------------------------------------
def _check_exclusion(treatment: str, policy_uin: str) -> tuple[bool, list[dict]]:
    """Return whether an exclusion clause names this treatment, and the clauses.

    Every retrieved exclusion clause is checked, not just the top few: an
    exclusion is the strongest finding the engine can make, so one ranked
    further down would otherwise be missed and the claim wrongly approved.
    """
    hits = check_exclusion.invoke({"treatment": treatment, "policy_uin": policy_uin})
    excluded = [hit for hit in hits if _mentions_treatment(hit["text"], treatment)]
    return bool(excluded), excluded[:2]


# ---------------------------------------------------------------------------
# Step 4: coverage evidence
# ---------------------------------------------------------------------------
# An indemnity health policy grants cover for hospitalisation in general, then
# names what it excludes, caps or defers. It does not enumerate the procedures
# it covers -- "appendectomy" appears nowhere in most wordings.
#
# So requiring the treatment to be named by name is not the safe reading, it is
# just a different wrong one: it parks ordinary, plainly covered claims as
# needs_more_info, which is the mirror image of the bug this rewrite set out to
# fix. What the engine needs is a clause that actually grants cover for what
# this claim is. These patterns match that grant.
#
# This is still positive evidence -- a retrieved clause saying the policy pays
# hospitalisation expenses -- and not the absence of a contrary clause, which
# is the distinction the whole three-state model turns on.
BASE_COVER_PATTERNS = (
    re.compile(r"hospitali[sz]ation expenses", re.I),
    re.compile(r"in-?patient (care|treatment|hospitali)", re.I),
    re.compile(r"medical expenses.{0,80}hospitali", re.I | re.S),
    re.compile(r"shall (indemnify|pay|reimburse).{0,120}hospitali", re.I | re.S),
    re.compile(r"expenses.{0,60}incurred.{0,60}hospitali", re.I | re.S),
)


def _grants_base_cover(text: str) -> bool:
    """True when a clause grants the policy's general hospitalisation cover."""
    return any(pattern.search(text or "") for pattern in BASE_COVER_PATTERNS)


def _check_coverage(treatment: str, policy_uin: str) -> PolicyFact:
    """Establish that the policy positively covers this treatment.

    Three kinds of evidence, strongest first: a curated record naming the
    treatment, a clause naming it, or the policy's base hospitalisation grant.
    If none of the three is there -- which means retrieval came back with
    nothing usable for this plan at all -- coverage is UNKNOWN and the claim
    goes to the employee. "No exclusion matched" is never one of the three.
    """
    store = get_policy_store()

    # A curated sub-limit row is a positive statement that the plan pays for
    # this treatment, and it is stronger evidence than any retrieved clause.
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
            f"'{treatment}' is not named individually, but this plan's "
            f"hospitalisation cover applies to it. Confirm the admission was "
            f"in-patient.",
            evidence=granting[:2],
        )

    return unknown(
        "coverage",
        f"No clause granting cover could be retrieved for this plan, so "
        f"whether '{treatment}' is covered is unestablished.",
    )


# ---------------------------------------------------------------------------
# Step 5: waiting period
# ---------------------------------------------------------------------------
def _waiting_period_fact(policy_uin: str, treatment: str) -> PolicyFact:
    """Resolve the waiting period: SQL, then the wording, then unknown.

    A curated schedule that lists other conditions but not this one is real
    evidence that this treatment has no waiting period -- that is
    NOT_APPLICABLE, not UNKNOWN. Retrieving nothing at all is UNKNOWN.
    """
    store = get_policy_store()

    row = store.find_waiting_period(policy_uin, treatment)
    if row is not None:
        return found(
            "waiting_period",
            row["waiting_period_months"],
            SOURCE_SQL,
            f"{row['waiting_period_months']}-month waiting period on record "
            f"for '{row.get('condition') or treatment}'.",
        )

    on_record = store.get_waiting_periods(policy_uin)
    if on_record:
        listed = ", ".join(r["condition"] for r in on_record if r.get("condition"))
        return not_applicable(
            "waiting_period",
            f"This plan's waiting-period schedule covers {listed} and does not "
            f"list '{treatment}'.",
        )

    terms = _specific_terms(treatment)
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
            return found(
                "waiting_period",
                months,
                SOURCE_WORDING,
                f"The wording states a {months}-month waiting period for "
                f"'{treatment}'.",
                evidence=[hit] if hit else [],
            )

    if hits and terms:
        return not_applicable(
            "waiting_period",
            f"The plan's waiting-period clauses do not name '{treatment}'.",
        )

    return unknown(
        "waiting_period",
        f"No waiting-period clause could be read for this plan, so whether "
        f"'{treatment}' has one is unestablished.",
    )


def _waiting_period_breach(policy: dict, fact: PolicyFact) -> str | None:
    """Return why the waiting period blocks the claim, or None if it does not."""
    if fact.status != FOUND:
        return None

    tracker = compute_waiting_period(policy["start_date"], fact.value)
    if tracker["is_eligible"]:
        return None
    return (
        f"Waiting period of {fact.value} months is not yet complete. "
        f"Cover begins {tracker['eligible_date']}."
    )


# ---------------------------------------------------------------------------
# Steps 6-8: the deduction facts
# ---------------------------------------------------------------------------
def _sub_limit_fact(policy_uin: str, treatment: str) -> PolicyFact:
    """Resolve the sub-limit: SQL, then the wording, then unknown."""
    row = get_policy_store().find_sub_limit(policy_uin, treatment)
    if row is not None:
        return found(
            "sub_limit",
            row["limit_amount"],
            SOURCE_SQL,
            f"Sub-limit of Rs {row['limit_amount']:,.0f} on record for "
            f"'{row.get('treatment') or treatment}'.",
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
            f"No clause caps '{treatment}', so the full bill is considered up "
            f"to the sum insured.",
        )

    return unknown(
        "sub_limit",
        f"No coverage wording could be retrieved, so any cap on '{treatment}' "
        f"is unestablished.",
    )


def _find_applicable_copay(copayments: list[dict], treatment: str) -> dict | None:
    """Prefer a co-pay row naming this treatment, else fall back to a blanket one.

    copayments is expected to already be narrowed to the customer's age by
    the caller (get_copayments(age=...)), so every row here is one that
    genuinely applies -- this only chooses between them by treatment name.

    If two age bands somehow overlap for the same age, this falls through to
    the first row the store returned. That is deterministic (SQLite returns
    the same order for the same data every time) but not a considered
    tie-break -- no policy in this corpus has overlapping bands to make one
    necessary, and inventing a "narrowest band wins" rule with nothing to
    test it against would be a guess dressed up as a feature.
    """
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
    """Render a copay row's age band as text, or "" when it has none."""
    age_min, age_max = row.get("age_min"), row.get("age_max")
    if age_min is not None and age_max is not None:
        return f"age {age_min}-{age_max}"
    if age_min is not None:
        return f"age {age_min}+"
    if age_max is not None:
        return f"age up to {age_max}"
    return ""


def _copay_fact(policy_uin: str, treatment: str, age: int | None) -> PolicyFact:
    """Resolve the co-pay percentage: SQL, then the wording, then unknown.

    age is the customer's age on the treatment date (None when it could not
    be determined -- see _customer_age_at_treatment). Some plans set a
    different percentage by age band (e.g. Future Generali's Health Total:
    20% at 60-64, rising to 40% at 75+); get_copayments(age=...) narrows to
    the rows that actually apply at that age. If a plan HAS age bands but the
    customer's age is unknown, guessing which band applies would be worse
    than saying so -- that case returns unknown rather than picking one.
    """
    store = get_policy_store()

    all_rows = store.get_copayments(policy_uin)
    is_age_banded = any(
        row.get("age_min") is not None or row.get("age_max") is not None
        for row in all_rows
    )
    if is_age_banded and age is None:
        return unknown(
            "copay",
            "This plan's co-payment depends on the customer's age, which "
            "could not be determined from the claim or the customer record.",
        )

    copayments = store.get_copayments(policy_uin, age=age)
    row = _find_applicable_copay(copayments, treatment)
    if row is not None:
        band = _age_band_text(row)
        return found(
            "copay",
            row["copay_percent"],
            SOURCE_SQL,
            f"Co-payment of {row['copay_percent']}% on record for "
            f"'{row.get('condition') or 'all claims'}'"
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
        return not_applicable(
            "copay", "No co-payment clause applies to this plan."
        )

    return unknown(
        "copay",
        "No co-payment wording could be retrieved, so the customer's share is "
        "unestablished.",
    )


def _deductible_fact(policy_uin: str) -> PolicyFact:
    """Resolve the deductible: SQL, then the wording, then unknown."""
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
        "No wording could be retrieved, so whether this plan carries a "
        "deductible is unestablished.",
    )


# ---------------------------------------------------------------------------
# Step 11: result shapes
# ---------------------------------------------------------------------------
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
    """Build the one result shape every caller reads."""
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
    """No payable figure is offered when a fact the decision rests on is missing."""
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
    """The customer's age on the treatment date, for age-banded co-pay.

    Deliberately the treatment date, not today: a claim assessed weeks after
    a birthday must still use the age the customer was when they were
    actually treated. Returns None -- not a guess -- when either the date of
    birth or the treatment date is missing, since an age-banded rule cannot
    be resolved without both.
    """
    customer = get_crm_store().get_customer(customer_id)
    date_of_birth = customer.get("date_of_birth") if customer else None
    if not date_of_birth or not treatment_date:
        return None
    return compute_age(date_of_birth, treatment_date)


# ---------------------------------------------------------------------------
# The checklist
# ---------------------------------------------------------------------------
def _policy_gate(
    policy: dict | None,
    customer_id: str,
    treatment: str,
    bill_amount: float,
    bill_data: dict,
) -> dict | None:
    """Steps 1-2, plus the claim's own completeness. None means carry on.

    The in-force check comes before the completeness check on purpose: a
    lapsed policy cannot pay whatever the claim says, so parking it as "we
    need more information" would send the employee chasing a detail that
    could not change the answer.
    """
    if policy is None or policy["customer_id"] != customer_id:
        return _result(
            INELIGIBLE, bill_amount,
            "This policy is not held by this customer.", [], [],
        )

    not_in_force = _check_policy_in_force(policy, bill_data.get("admission_date"))
    if not_in_force:
        return _result(INELIGIBLE, bill_amount, not_in_force, [], [])

    # A claim that names no procedure cannot be assessed against a policy that
    # states its conditions per procedure. The gap is in the claim, not in the
    # policy, so it is reported as such.
    if not _specific_terms(treatment):
        return _needs_more_info(
            bill_amount, [], [],
            f"The claim gives '{treatment or 'no treatment'}', which does not "
            f"name a specific procedure or condition to assess.",
            ["A specific treatment or diagnosis on the claim form."],
        )
    return None


def _clause_checks(
    policy: dict, treatment: str, bill_amount: float
) -> tuple[dict | None, PolicyFact | None, PolicyFact | None, list[dict]]:
    """Steps 3-5: the checks that can end a claim before any arithmetic.

    Returns (early_result, coverage, waiting, evidence). A non-None first
    element means the claim is settled and the rest are meaningless.
    """
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
            f"Cover for '{treatment}' could not be confirmed from this "
            f"policy's wording or records.",
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
    policy_uin: str, treatment: str, bill_amount: float, base: list[PolicyFact],
    evidence: list[dict], age: int | None,
) -> tuple[dict | None, list[PolicyFact]]:
    """Steps 6-8. An unresolved fact stops the claim before any arithmetic."""
    sub_limit = _sub_limit_fact(policy_uin, treatment)
    copay = _copay_fact(policy_uin, treatment, age)
    deductible = _deductible_fact(policy_uin)
    facts = [*base, sub_limit, copay, deductible]

    unresolved = [fact for fact in facts if not fact.is_known]
    if unresolved:
        return _needs_more_info(
            bill_amount, facts, evidence,
            "The payable amount cannot be calculated until "
            + ", ".join(fact.name.replace("_", " ") for fact in unresolved)
            + " is established.",
            [fact.detail for fact in unresolved],
        ), facts

    for fact in (sub_limit, copay, deductible):
        evidence.extend(fact.evidence)
    return None, facts


def _price_the_claim(
    policy_id: str, customer_id: str, treatment: str, bill_amount: float,
    facts: list[PolicyFact], evidence: list[dict],
) -> dict:
    """Steps 9-10: what is left of the cover, then the arithmetic."""
    sub_limit, copay, deductible = facts[-3], facts[-2], facts[-1]

    balance = compute_sum_insured_balance(policy_id, customer_id)
    remaining = balance.get("remaining_balance", 0)
    facts = [
        *facts,
        found(
            "remaining_sum_insured", remaining, SOURCE_SQL,
            f"Rs {remaining:,.0f} of Rs {balance.get('sum_insured', 0):,.0f} "
            f"remains for this policy year.",
        ),
    ]
    if remaining <= 0:
        return _result(
            INELIGIBLE, bill_amount,
            "The sum insured for this policy year is exhausted.",
            evidence, facts,
        )

    # The one place a rupee figure is produced, and it is a pure function.
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
        f"'{treatment}' is covered, no exclusion applies, and every condition "
        f"on it was checked.",
        evidence,
        facts,
        covered_amount=bill_amount - payable["deductions"]["sub_limit_reduction"],
        copay_amount=payable["deductions"]["copay_amount"],
        payable=payable["payable_amount"],
        deductions=payable["deductions"],
    )


def check_eligibility(bill_data: dict, customer_id: str, policy_id: str) -> dict:
    """Run the full eligibility checklist for one claim against one policy.

    Each phase either settles the claim and returns a result, or hands the
    next one what it established. Nothing is assumed on the way through: a
    fact that could not be read stops the claim rather than defaulting.
    """
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
        policy["policy_number"], treatment, bill_amount,
        [coverage, waiting], evidence, age,
    )
    if unresolved:
        return unresolved

    return _price_the_claim(
        policy_id, customer_id, treatment, bill_amount, facts, evidence
    )
