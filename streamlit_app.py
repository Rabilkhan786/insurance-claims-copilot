"""Claims Copilot - the employee-facing frontend.

Run with:  uv run streamlit run streamlit_app.py

This is an INTERNAL tool. The person using it is a claims employee, not a
customer. The screen shows a recommendation and the employee decides: the
Approve / Edit / Reject actions at the bottom are the point of the whole app.
"""
from __future__ import annotations

import sqlite3
from datetime import date
from uuid import uuid4

import streamlit as st

st.set_page_config(
    page_title="Claims Copilot",
    page_icon="clipboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading retrieval models - first run only...")
def _warmup():
    """Build the embedder and reranker once per process, not per click."""
    from src.tools.rag_tools import warmup

    warmup()
    return True


@st.cache_data(ttl=60)
def _customers() -> list[dict]:
    """List real customers from the CRM so no IDs are hardcoded here."""
    from config import settings

    try:
        connection = sqlite3.connect(settings.crm_db_path)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT customer_id, name FROM customers ORDER BY customer_id"
        ).fetchall()
        connection.close()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


@st.cache_data(ttl=60)
def _policies_for(customer_id: str) -> list[dict]:
    """Policies belonging to one customer, for the policy dropdown."""
    from config import settings

    try:
        connection = sqlite3.connect(settings.crm_db_path)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT policy_id, policy_name, policy_number, sum_insured "
            "FROM policies WHERE customer_id = ? ORDER BY policy_id",
            (customer_id,),
        ).fetchall()
        connection.close()
        return [dict(row) for row in rows]
    except sqlite3.Error:
        return []


def _init_state() -> None:
    """Session state: one pending review at a time."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "review" not in st.session_state:
        st.session_state.review = None
    if "saved" not in st.session_state:
        st.session_state.saved = None


def _money(value) -> str:
    """Format a rupee amount, tolerating None."""
    if value is None:
        return "-"
    return f"Rs {float(value):,.2f}"


def _run_analysis(claim: dict, customer_id: str, policy_id: str) -> None:
    """Call the workflow and park the result for the review panel."""
    from src.agent import run_claim_review

    # A fresh thread per analysis: the graph pauses on this id, and resuming
    # the wrong one would attach the decision to an older claim.
    st.session_state.session_id = str(uuid4())
    st.session_state.saved = None

    with st.spinner("Running eligibility checks and retrieving policy clauses..."):
        result = run_claim_review(
            claim, customer_id, policy_id, st.session_state.session_id
        )

    if result.get("error"):
        st.error(result["error"])
        st.session_state.review = None
        return

    result["claim"] = claim
    st.session_state.review = result


def render_claim_form() -> None:
    """The claim-entry form. Submitting it runs the whole analysis."""
    customers = _customers()
    if not customers:
        st.error("No customers found in the CRM. Run `python Data/seed.py` first.")
        return

    labels = {f"{c['customer_id']} - {c['name']}": c["customer_id"] for c in customers}
    chosen = st.selectbox("Customer", list(labels))
    customer_id = labels[chosen]

    policies = _policies_for(customer_id)
    if not policies:
        st.warning("This customer has no policies on file.")
        return

    policy_labels = {
        f"{p['policy_id']} - {p['policy_name']} (SI {_money(p['sum_insured'])})":
        p["policy_id"]
        for p in policies
    }
    chosen_policy = st.selectbox("Policy", list(policy_labels))
    policy_id = policy_labels[chosen_policy]

    with st.form("claim_form"):
        left, right = st.columns(2)
        with left:
            claim_amount = st.number_input(
                "Claim amount (Rs)", min_value=1.0, value=50000.0, step=1000.0
            )
            treatment = st.text_input("Treatment / procedure", "Cataract Surgery")
            diagnosis = st.text_input("Diagnosis", "")
        with right:
            hospital = st.text_input("Hospital", "")
            treatment_date = st.date_input("Date of treatment", value=date.today())
            claim_id = st.text_input("Claim ID", f"CLM-{uuid4().hex[:8].upper()}")

        notes = st.text_area("Notes for the file", "")
        submitted = st.form_submit_button("Analyze claim", type="primary")

    if submitted:
        _run_analysis(
            {
                "claim_id": claim_id,
                "claim_amount": claim_amount,
                "treatment": treatment,
                "diagnosis": diagnosis,
                "hospital": hospital,
                "treatment_date": str(treatment_date),
                "notes": notes,
            },
            customer_id,
            policy_id,
        )


def render_breakdown(eligibility: dict) -> None:
    """Show the deduction breakdown the engine produced."""
    first, second, third = st.columns(3)
    first.metric("Bill amount", _money(eligibility.get("bill_amount")))
    second.metric("Co-pay deducted", _money(eligibility.get("copay_amount")))
    third.metric("Recommended payable", _money(eligibility.get("estimated_payable")))

    if eligibility.get("rejection_reason"):
        st.warning(f"Deciding check: {eligibility['rejection_reason']}")


def render_evidence(eligibility: dict) -> None:
    """List the cited policy clauses behind the recommendation."""
    evidence = eligibility.get("evidence") or []
    if not evidence:
        st.info("No policy clauses were retrieved for this claim.")
        return

    st.markdown("**Cited evidence**")
    for item in evidence:
        uin = item.get("uin") or "-"
        page = item.get("page") or "-"
        insurer = item.get("insurer") or "-"
        with st.expander(f"[Source: {insurer}, UIN: {uin}, Page {page}]"):
            st.write(item.get("text") or "")


def _save(decision: str, employee: str, **extra) -> None:
    """Resume the paused graph with the employee's answer and record it."""
    from src.agent import submit_decision

    with st.spinner("Recording decision..."):
        result = submit_decision(
            st.session_state.session_id, decision, employee, **extra
        )

    if result.get("error"):
        st.error(result["error"])
        return

    st.session_state.saved = decision
    st.session_state.review = None
    st.rerun()


def _render_actions(employee: str, eligibility: dict) -> None:
    """Approve as-is, approve with edits, or reject the recommendation."""
    approve, edit, reject = st.tabs(["Approve", "Edit", "Reject"])

    with approve:
        st.write("Record the recommendation exactly as it stands.")
        if st.button("Approve as recommended", type="primary"):
            _save("approve", employee)

    with edit:
        amount = st.number_input(
            "Corrected payable amount (Rs)",
            min_value=0.0,
            value=float(eligibility.get("estimated_payable") or 0),
            step=500.0,
        )
        edits = st.text_area("What you changed and why", key="edit_note")
        if st.button("Save edited decision"):
            _save("edit", employee, payable_amount=amount, edits=edits)

    with reject:
        reason = st.text_area("Reason for rejecting", key="reject_note")
        if st.button("Reject recommendation"):
            if not reason.strip():
                st.warning("Please give a reason before rejecting.")
            else:
                _save("reject", employee, reason=reason)


def render_review_panel(employee: str) -> None:
    """The recommendation plus the three actions the employee can take."""
    review = st.session_state.review
    if not review:
        return

    eligibility = review.get("eligibility") or {}
    st.divider()
    st.subheader("Recommendation - awaiting your review")

    if eligibility.get("eligible"):
        st.success("Engine result: eligible")
    else:
        st.error("Engine result: not eligible")

    render_breakdown(eligibility)
    st.markdown(review.get("recommendation") or "")
    render_evidence(eligibility)

    st.divider()
    st.markdown("**Your decision** - this is what gets recorded, not the AI's.")
    _render_actions(employee, eligibility)


def render_audit_trail() -> None:
    """Recent decisions, so the employee can see what has been recorded."""
    from config import settings
    from src.decisions import DecisionStore

    store = DecisionStore(settings.crm_db_path)
    rows = store.recent(limit=10)
    stats = store.agreement_rate()

    st.caption(
        f"{stats['total']} decisions recorded, "
        f"{stats['agreement_rate_percent']}% approved as recommended"
    )
    if not rows:
        st.info("No decisions recorded yet.")
        return

    st.dataframe(
        [
            {
                "Claim": row["claim_id"],
                "AI said": row["ai_decision"],
                "Employee": row["employee_decision"],
                "AI amount": row["ai_payable_amount"],
                "Final amount": (
                    row["employee_payable_amount"] or row["ai_payable_amount"]
                ),
                "By": row["decided_by"],
            }
            for row in rows
        ],
    )


def main() -> None:
    """Draw the whole page: form on the left, audit trail in the sidebar."""
    _init_state()
    st.title("Claims Copilot")
    st.caption(
        "Internal tool. The copilot recommends, you decide. "
        "Every recommendation is stored alongside your decision."
    )

    with st.sidebar:
        st.header("Reviewer")
        employee = st.text_input("Your employee ID", "emp.demo")
        st.divider()
        st.header("Audit trail")
        render_audit_trail()

    _warmup()

    if st.session_state.saved:
        st.success(f"Decision recorded: {st.session_state.saved}")

    render_claim_form()
    render_review_panel(employee)


main()
