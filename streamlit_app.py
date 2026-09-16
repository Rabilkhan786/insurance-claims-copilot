"""Streamlit interface for claims employees."""
from __future__ import annotations

from datetime import date
from uuid import uuid4

import streamlit as st

st.set_page_config(
    page_title="Claims Copilot",
    page_icon="clipboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

STATUS_DISPLAY = {
    "approve": ("success", "Recommend: approve"),
    "reject": ("error", "Recommend: reject"),
    "needs_more_info": ("warning", "Recommend: needs more info"),
}

FACT_STATUS_DISPLAY = {
    "found": "established",
    "not_applicable": "does not apply",
    "unknown": "NOT ESTABLISHED",
}


@st.cache_resource(show_spinner="Loading retrieval models - first run only...")
def _warmup():
    """Load retrieval models once per Streamlit process."""
    from src.tools.rag_tools import warmup

    warmup()
    return True


@st.cache_data(ttl=60)
def _customers() -> list[dict]:
    """Return customers for the UI picker."""
    from src.crm import get_crm_store

    return get_crm_store().list_customers()


@st.cache_data(ttl=60)
def _policies_for(customer_id: str) -> list[dict]:
    """Return one customer's policies."""
    from src.crm import get_crm_store

    return get_crm_store().get_policies(customer_id)


def _init_state() -> None:
    """Initialize claim-review and chat session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "review" not in st.session_state:
        st.session_state.review = None
    if "saved" not in st.session_state:
        st.session_state.saved = None
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _money(value) -> str:
    """Format a rupee amount without treating None as zero."""
    if value is None:
        return "not calculated"
    return f"Rs {float(value):,.2f}"


def _run_analysis(claim: dict, customer_id: str, policy_id: str) -> None:
    """Run claim analysis and store the pending review."""
    from src.agent import run_claim_review

    st.session_state.session_id = str(uuid4())
    st.session_state.saved = None

    with st.spinner("Running eligibility checks and retrieving policy clauses..."):
        result = run_claim_review(
            claim,
            customer_id,
            policy_id,
            st.session_state.session_id,
        )

    if result.get("error"):
        st.error(result["error"])
        st.session_state.review = None
        return

    result["claim"] = claim
    st.session_state.review = result


def render_claim_form() -> None:
    """Render the employee claim-entry form."""
    customers = _customers()
    if not customers:
        st.error("No customers found. Run `uv run python Data/seed.py` first.")
        return

    labels = {
        f"{customer['customer_id']} - {customer['name']}": customer["customer_id"]
        for customer in customers
    }
    chosen = st.selectbox("Customer", list(labels))
    customer_id = labels[chosen]

    policies = _policies_for(customer_id)
    if not policies:
        st.warning("This customer has no policies on file.")
        return

    policy_labels = {
        (
            f"{policy['policy_id']} - {policy['policy_name']} "
            f"(SI {_money(policy['sum_insured'])})"
        ): policy["policy_id"]
        for policy in policies
    }
    chosen_policy = st.selectbox("Policy", list(policy_labels))
    policy_id = policy_labels[chosen_policy]

    with st.form("claim_form"):
        left, right = st.columns(2)

        with left:
            claim_amount = st.number_input(
                "Claim amount (Rs)",
                min_value=1.0,
                value=50000.0,
                step=1000.0,
            )
            treatment = st.text_input("Treatment / procedure", "Cataract Surgery")
            diagnosis = st.text_input("Diagnosis", "")

        with right:
            hospital = st.text_input("Hospital", "")
            treatment_date = st.date_input("Date of treatment", value=date.today())
            claim_id = st.text_input(
                "Claim ID",
                f"CLM-{uuid4().hex[:8].upper()}",
            )

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


def render_chat() -> None:
    """Render the customer-scoped policy and claims chat."""
    from src.agent import run_agent

    customers = _customers()
    if not customers:
        return

    st.subheader("Ask a question")
    labels = {
        f"{customer['customer_id']} - {customer['name']}": customer["customer_id"]
        for customer in customers
    }
    chosen = st.selectbox(
        "About which customer?",
        list(labels),
        key="chat_customer",
    )
    customer_id = labels[chosen]

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Question",
            placeholder="e.g. What is this customer's remaining sum insured?",
        )
        asked = st.form_submit_button("Ask")

    if asked and question.strip():
        with st.spinner("Thinking..."):
            result = run_agent(
                question,
                st.session_state.chat_session_id,
                customer_id,
            )
        answer = result.get("error") or result.get("answer") or "No answer."
        st.session_state.chat_history.append((question, answer))

    for asked_question, answer in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(asked_question)
        with st.chat_message("assistant"):
            st.write(answer)


def render_breakdown(recommendation: dict) -> None:
    """Show claim amount, deductions, and recommended payable amount."""
    deductions = recommendation.get("deductions") or {}
    first, second, third = st.columns(3)

    first.metric("Bill amount", _money(recommendation.get("bill_amount")))
    second.metric(
        "Co-pay deducted",
        _money(deductions.get("copay_amount") or 0),
    )
    third.metric(
        "Recommended payable",
        _money(recommendation.get("payable_amount")),
    )

    if recommendation.get("reason_summary"):
        st.caption(f"Deciding check: {recommendation['reason_summary']}")


def _fact_value(value) -> str:
    """Convert a fact value to a display-safe string."""
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def render_facts(recommendation: dict) -> None:
    """Show policy facts resolved by the eligibility engine."""
    facts = recommendation.get("facts") or []
    if not facts:
        return

    st.markdown("**What the engine established**")
    st.dataframe(
        [
            {
                "Fact": fact["name"].replace("_", " ").title(),
                "Status": FACT_STATUS_DISPLAY.get(
                    fact["status"],
                    fact["status"],
                ),
                "Value": _fact_value(fact.get("value")),
                "Source": fact.get("source", "-"),
                "Detail": fact.get("detail", ""),
            }
            for fact in facts
        ],
        hide_index=True,
    )


def render_missing(recommendation: dict) -> None:
    """Show information still needed before a decision can be made."""
    missing = recommendation.get("missing_information") or []
    if not missing:
        return

    st.warning("**Missing information** - chase these before deciding:")
    for item in missing:
        st.markdown(f"- {item}")


def render_evidence(recommendation: dict) -> None:
    """Show the policy clauses supporting the recommendation."""
    evidence = recommendation.get("evidence") or []
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
    """Resume the paused workflow and save the employee decision."""
    from src.agent import submit_decision

    with st.spinner("Recording decision..."):
        result = submit_decision(
            st.session_state.session_id,
            decision,
            employee,
            **extra,
        )

    if result.get("error"):
        st.error(result["error"])
        return

    st.session_state.saved = decision
    st.session_state.review = None
    st.rerun()


def _render_actions(employee: str, recommendation: dict) -> None:
    """Render approve, edit, and reject actions."""
    approve, edit, reject = st.tabs(["Approve", "Edit", "Reject"])
    payable = recommendation.get("payable_amount")

    with approve:
        if recommendation.get("status") == "needs_more_info":
            st.info(
                "The copilot could not establish every required fact. "
                "Approving records that you decided anyway."
            )
        st.write("Record the recommendation exactly as it stands.")
        if st.button("Approve as recommended", type="primary"):
            _save("approve", employee)

    with edit:
        amount = st.number_input(
            "Corrected payable amount (Rs)",
            min_value=0.0,
            value=float(payable or 0),
            step=500.0,
        )
        edits = st.text_area("What you changed and why", key="edit_note")
        if st.button("Save edited decision"):
            _save(
                "edit",
                employee,
                payable_amount=amount,
                edits=edits,
            )

    with reject:
        reason = st.text_area("Reason for rejecting", key="reject_note")
        if st.button("Reject recommendation"):
            if not reason.strip():
                st.warning("Please give a reason before rejecting.")
            else:
                _save("reject", employee, reason=reason)


def render_review_panel(employee: str) -> None:
    """Show the recommendation and employee decision controls."""
    review = st.session_state.review
    if not review:
        return

    recommendation = review.get("recommendation") or {}
    st.divider()
    st.subheader("Recommendation - awaiting your review")

    style, label = STATUS_DISPLAY.get(
        recommendation.get("status"),
        ("info", "No recommendation"),
    )
    getattr(st, style)(label)

    render_breakdown(recommendation)
    render_missing(recommendation)
    st.markdown(recommendation.get("reasoning") or "")
    render_facts(recommendation)
    render_evidence(recommendation)

    st.divider()
    st.markdown("**Your decision** - this is what gets recorded, not the AI's.")
    _render_actions(employee, recommendation)


def render_audit_trail() -> None:
    """Show recent reviewed claim decisions."""
    from config import settings
    from src.decisions import DecisionStore

    store = DecisionStore(settings.crm_db_path)
    rows = store.recent(limit=10)
    stats = store.agreement_rate()

    st.caption(
        f"{stats['total']} decisions recorded, "
        f"{stats['agreement_rate_percent']}% matched the AI recommendation"
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
                    row["employee_payable_amount"]
                    if row["employee_payable_amount"] is not None
                    else row["ai_payable_amount"]
                ),
                "By": row["decided_by"],
            }
            for row in rows
        ],
        hide_index=True,
    )


def main() -> None:
    """Render the complete employee interface."""
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

    render_chat()
    st.divider()
    render_claim_form()
    render_review_panel(employee)


main()
