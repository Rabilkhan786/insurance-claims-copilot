"""Claims Copilot - the employee-facing frontend.

Run with:  uv run streamlit run streamlit_app.py

This is an INTERNAL tool. The person using it is a claims employee, not a
customer. The screen shows a recommendation and the employee decides: the
Approve / Edit / Reject actions at the bottom are the point of the whole app.

There is no business logic in this file. Every figure on screen came from the
deterministic engine, every clause from the retrieval layer, and every
database read goes through the CRM store -- the UI only lays them out.
"""
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

# How the engine's three states read on screen.
STATUS_DISPLAY = {
    "approve": ("success", "Recommend: approve"),
    "reject": ("error", "Recommend: reject"),
    "needs_more_info": ("warning", "Recommend: needs more info"),
}

# A fact's status, in words an employee reads rather than a field value.
FACT_STATUS_DISPLAY = {
    "found": "established",
    "not_applicable": "does not apply",
    "unknown": "NOT ESTABLISHED",
}


@st.cache_resource(show_spinner="Loading retrieval models - first run only...")
def _warmup():
    """Build the embedder and reranker once per process, not per click."""
    from src.tools.rag_tools import warmup

    warmup()
    return True


@st.cache_data(ttl=60)
def _customers() -> list[dict]:
    """List real customers from the CRM so no IDs are hardcoded here."""
    from src.crm import get_crm_store

    return get_crm_store().list_customers()


@st.cache_data(ttl=60)
def _policies_for(customer_id: str) -> list[dict]:
    """Policies belonging to one customer, for the policy dropdown."""
    from src.crm import get_crm_store

    return get_crm_store().get_policies(customer_id)


def _init_state() -> None:
    """Session state: one pending review at a time, plus a running chat."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "review" not in st.session_state:
        st.session_state.review = None
    if "saved" not in st.session_state:
        st.session_state.saved = None
    # A separate thread from the claim review above: chat and claim analysis
    # are two different graph runs, and sharing one session_id would mix a
    # customer's chat history into the claim's paused review state.
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = str(uuid4())
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def _money(value) -> str:
    """Format a rupee amount, tolerating None.

    None is not zero: it means the engine declined to give a figure because a
    fact it depends on is missing. Showing "Rs 0.00" there would read as
    "nothing is payable", which is a different and much stronger claim.
    """
    if value is None:
        return "not calculated"
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
        st.error("No customers found in the CRM. Run `uv run python Data/seed.py` first.")
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


def render_chat() -> None:
    """A general Q&A box -- separate from the claim form above.

    Not every question an employee has is "assess this claim" -- sometimes
    it's just "what's this customer's remaining sum insured". This calls the
    same agent that handles that, with memory: it remembers earlier turns in
    this chat because every call reuses one chat_session_id.
    """
    from src.agent import run_agent

    customers = _customers()
    if not customers:
        return

    st.subheader("Ask a question")
    labels = {f"{c['customer_id']} - {c['name']}": c["customer_id"] for c in customers}
    chosen = st.selectbox("About which customer?", list(labels), key="chat_customer")
    customer_id = labels[chosen]

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input(
            "Question", placeholder="e.g. What is this customer's remaining sum insured?"
        )
        asked = st.form_submit_button("Ask")

    if asked and question.strip():
        with st.spinner("Thinking..."):
            result = run_agent(
                question, st.session_state.chat_session_id, customer_id
            )
        answer = result.get("error") or result.get("answer") or "No answer."
        st.session_state.chat_history.append((question, answer))

    for asked_question, answer in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(asked_question)
        with st.chat_message("assistant"):
            st.write(answer)


def render_breakdown(recommendation: dict) -> None:
    """Show the deduction breakdown the engine produced."""
    deductions = recommendation.get("deductions") or {}
    first, second, third = st.columns(3)
    first.metric("Bill amount", _money(recommendation.get("bill_amount")))
    second.metric("Co-pay deducted", _money(deductions.get("copay_amount") or 0))
    third.metric("Recommended payable", _money(recommendation.get("payable_amount")))

    if recommendation.get("reason_summary"):
        st.caption(f"Deciding check: {recommendation['reason_summary']}")


def _fact_value(value) -> str:
    """Render one fact's value as a single string.

    A fact's value is a bool (coverage), a float (sub_limit, copay,
    remaining sum insured), or None (not_applicable / unknown) depending on
    which fact it is. Handing that mix straight to st.dataframe put all
    three in one column; PyArrow inferred a boolean column from the
    True/False rows, then failed converting the floats and "-" placeholders
    into it. Streamlit caught the exception and silently coerced the table,
    but every render was throwing a full traceback into the server log.
    A single string type sidesteps the inference entirely.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def render_facts(recommendation: dict) -> None:
    """Show every policy fact the engine resolved, and how it resolved it.

    This table is what separates a copilot from a black box: an employee can
    see that the co-pay came from the policy records and the sub-limit from
    the wording, and that nothing was assumed.
    """
    facts = recommendation.get("facts") or []
    if not facts:
        return

    st.markdown("**What the engine established**")
    st.dataframe(
        [
            {
                "Fact": fact["name"].replace("_", " ").title(),
                "Status": FACT_STATUS_DISPLAY.get(fact["status"], fact["status"]),
                "Value": _fact_value(fact.get("value")),
                "Source": fact.get("source", "-"),
                "Detail": fact.get("detail", ""),
            }
            for fact in facts
        ],
        hide_index=True,
    )


def render_missing(recommendation: dict) -> None:
    """List what the employee has to chase before this claim can be decided."""
    missing = recommendation.get("missing_information") or []
    if not missing:
        return

    st.warning("**Missing information** - chase these before deciding:")
    for item in missing:
        st.markdown(f"- {item}")


def render_evidence(recommendation: dict) -> None:
    """List the cited policy clauses behind the recommendation."""
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


def _render_actions(employee: str, recommendation: dict) -> None:
    """Approve as-is, approve with edits, or reject the recommendation."""
    approve, edit, reject = st.tabs(["Approve", "Edit", "Reject"])
    payable = recommendation.get("payable_amount")

    with approve:
        if recommendation.get("status") == "needs_more_info":
            st.info(
                "The copilot could not establish everything this claim turns "
                "on. Approving records that you decided anyway."
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

    recommendation = review.get("recommendation") or {}
    st.divider()
    st.subheader("Recommendation - awaiting your review")

    style, label = STATUS_DISPLAY.get(
        recommendation.get("status"), ("info", "No recommendation")
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
        hide_index=True,
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

    render_chat()
    st.divider()
    render_claim_form()
    render_review_panel(employee)


main()
