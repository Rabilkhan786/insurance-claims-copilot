"""LangGraph workflow for claim review and employee chat."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt

from config import settings
from src.agent.agent import Context, get_agent, get_checkpointer, get_claims_agent
from src.agent.recommendation import ClaimRecommendation
from src.decisions import DecisionStore
from src.eligibility import check_eligibility

logger = logging.getLogger(__name__)


class WorkflowState(MessagesState, total=False):
    """MessagesState plus fields used by the claim-review path."""

    claim: dict[str, Any] | None
    claim_id: str | None
    policy_id: str | None
    customer_id: str | None
    eligibility: dict[str, Any] | None
    structured_response: Any | None
    recommendation: dict[str, Any] | None
    decision: dict[str, Any] | None
    error: str | None


WRITE_RECOMMENDATION_PROMPT = """Write a claim recommendation for the claims employee reviewing this file. The figures below are final -- do not recalculate them, do not round them, and do not introduce any amount that is not listed.

Structure it as:
1. Recommend: approve / reject / needs more info -- and why, in one sentence.
2. The payable amount, and which deduction reduced the bill to it. If the status is needs_more_info there is no payable amount yet -- say what has to be established before one can be calculated, and do not invent a figure.
3. The policy clauses that support the decision, each with its citation in the exact format [Source: {{insurer}}, UIN: {{uin}}, Page {{page}}].
4. Anything missing that the employee should chase before deciding.

Read the per-fact statuses before you write. Only "unknown" belongs under "Missing information" or as something you "could not find". A "not_applicable" fact means the policy was checked and confirmed to state no such condition; report it as a finding rather than as something absent.

This is a recommendation for a human to review, never a final answer. Do not write "the claim is approved" -- write "recommend: approve".

Engine result:
{eligibility}

Claim as submitted:
{claim}"""


@lru_cache(maxsize=1)
def get_decision_store() -> DecisionStore:
    """Return the shared audit-trail store."""
    return DecisionStore(settings.crm_db_path)


def _to_bill_data(claim: dict) -> dict:
    """Map employee form fields to the eligibility engine schema."""
    treatment = claim.get("treatment")
    if treatment is None or treatment == "":
        treatment = claim.get("procedure") or ""

    total_amount = claim.get("claim_amount")
    if total_amount is None:
        total_amount = claim.get("total_amount")
    if total_amount is None:
        total_amount = 0

    admission_date = claim.get("treatment_date")
    if admission_date is None:
        admission_date = claim.get("admission_date")

    return {
        "treatment": treatment,
        "diagnosis": claim.get("diagnosis") or "",
        "total_amount": total_amount,
        "hospital": claim.get("hospital") or "",
        "admission_date": admission_date,
    }


def eligibility_node(state: WorkflowState, runtime: Runtime[Context]) -> dict:
    """Run deterministic eligibility and prepare the explanation prompt."""
    customer_id = runtime.context.customer_id
    policy_id = state.get("policy_id")
    if not customer_id or not policy_id:
        return {"error": "A customer ID and policy ID are both needed to check a claim."}

    claim = state.get("claim") or {}
    bill_data = _to_bill_data(claim)
    logger.info(
        "eligibility_check customer_id=%s policy_id=%s treatment=%r",
        customer_id,
        policy_id,
        bill_data["treatment"],
    )

    eligibility = check_eligibility(bill_data, customer_id, policy_id)
    logger.info(
        "eligibility_result customer_id=%s policy_id=%s status=%s payable=%s",
        customer_id,
        policy_id,
        eligibility.get("status"),
        eligibility.get("estimated_payable"),
    )

    return {
        "eligibility": eligibility,
        "customer_id": customer_id,
        "messages": [
            HumanMessage(
                WRITE_RECOMMENDATION_PROMPT.format(
                    eligibility=eligibility,
                    claim=claim,
                )
            )
        ],
    }


def _explanation_text(state: WorkflowState) -> str:
    """Read structured explanation output, with a test-friendly fallback."""
    structured = state.get("structured_response")
    if structured is not None:
        return _normalise_citations(structured.reasoning)
    return _last_ai_text(state.get("messages", []))


def review_node(state: WorkflowState) -> dict:
    """Pause for employee review and return the selected decision on resume."""
    recommendation = ClaimRecommendation.from_engine(
        state.get("eligibility") or {},
        reasoning=_explanation_text(state),
    )
    logger.info("review_waiting status=%s", recommendation.status)

    decision = interrupt(
        {
            "recommendation": recommendation.model_dump(),
            "claim": state.get("claim"),
        }
    )

    logger.info("review_completed decision=%s", decision.get("decision"))
    return {
        "recommendation": recommendation.model_dump(),
        "decision": decision,
    }


def persist_decision_node(state: WorkflowState) -> dict:
    """Persist the recommendation and the employee decision in the audit trail."""
    decision = state.get("decision") or {}
    claim = state.get("claim") or {}

    try:
        get_decision_store().record(
            claim_id=state.get("claim_id") or claim.get("claim_id") or str(uuid4()),
            customer_id=state.get("customer_id") or "",
            policy_id=state.get("policy_id"),
            recommendation=state.get("recommendation") or {},
            employee_decision=decision.get("decision", "approve"),
            employee_payable_amount=decision.get("payable_amount"),
            employee_edits=decision.get("edits"),
            override_reason=decision.get("reason"),
            decided_by=decision.get("decided_by") or "unknown",
            notes=claim.get("notes"),
        )
    except Exception:
        logger.exception("audit_write_failed claim_id=%s", state.get("claim_id"))
        return {"error": "The decision was made but could not be written to the audit trail."}

    return {}


def _route_entry(state: WorkflowState) -> str:
    return "eligibility" if state.get("claim") else "agent"


def _after_eligibility(state: WorkflowState) -> str:
    return END if state.get("error") else "explain_claim"


_workflow = None


def get_workflow():
    """Return the compiled outer workflow, building it once per process."""
    global _workflow
    if _workflow is None:
        builder = StateGraph(WorkflowState, context_schema=Context)
        builder.add_node("eligibility", eligibility_node)
        builder.add_node("agent", get_agent())
        builder.add_node("explain_claim", get_claims_agent())
        builder.add_node("review", review_node)
        builder.add_node("persist_decision", persist_decision_node)

        builder.add_conditional_edges(
            START,
            _route_entry,
            {"eligibility": "eligibility", "agent": "agent"},
        )
        builder.add_conditional_edges(
            "eligibility",
            _after_eligibility,
            {"explain_claim": "explain_claim", END: END},
        )
        builder.add_edge("explain_claim", "review")
        builder.add_edge("agent", END)
        builder.add_edge("review", "persist_decision")
        builder.add_edge("persist_decision", END)
        _workflow = builder.compile(checkpointer=get_checkpointer())
        logger.info("workflow_built")
    return _workflow


def _normalise_citations(text: str) -> str:
    """Normalize full-width citation brackets to plain ASCII brackets."""
    return text.replace("【", "[").replace("】", "]")


def _last_ai_text(messages: list) -> str:
    """Return the final non-empty assistant message."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return _normalise_citations(message.content)
    return "I could not find this in your policy documents."


def _thread_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _interrupt_payload(state: dict) -> dict | None:
    interrupts = state.get("__interrupt__")
    if not interrupts:
        return None
    return dict(interrupts[0].value)


def run_claim_review(
    claim: dict,
    customer_id: str,
    policy_id: str,
    session_id: str | None = None,
) -> dict:
    """Analyse a claim and pause for employee review."""
    session_id = session_id or str(uuid4())
    try:
        logger.info(
            "claim_review session_id=%s customer_id=%s policy_id=%s",
            session_id,
            customer_id,
            policy_id,
        )
        state = get_workflow().invoke(
            {
                "claim": claim,
                "claim_id": claim.get("claim_id"),
                "policy_id": policy_id,
            },
            config=_thread_config(session_id),
            context=Context(customer_id=customer_id),
        )

        pending = _interrupt_payload(state)
        if pending is None:
            return {
                "session_id": session_id,
                "awaiting_review": False,
                "error": state.get("error") or "The claim could not be analysed.",
            }

        return {
            "session_id": session_id,
            "awaiting_review": True,
            "recommendation": pending.get("recommendation"),
            "error": None,
        }
    except Exception:
        logger.exception("claim_review_failed session_id=%s", session_id)
        return {
            "session_id": session_id,
            "awaiting_review": False,
            "error": "Something went wrong analysing this claim. Please try again.",
        }


def submit_decision(
    session_id: str,
    decision: str,
    decided_by: str,
    payable_amount: float | None = None,
    edits: str | None = None,
    reason: str | None = None,
) -> dict:
    """Resume a paused review with the employee's decision."""
    try:
        logger.info(
            "decision_submitted session_id=%s decision=%s by=%s",
            session_id,
            decision,
            decided_by,
        )
        state = get_workflow().invoke(
            Command(
                resume={
                    "decision": decision,
                    "decided_by": decided_by,
                    "payable_amount": payable_amount,
                    "edits": edits,
                    "reason": reason,
                }
            ),
            config=_thread_config(session_id),
        )
        return {
            "session_id": session_id,
            "recorded": not state.get("error"),
            "error": state.get("error"),
        }
    except Exception:
        logger.exception("decision_submit_failed session_id=%s", session_id)
        return {
            "session_id": session_id,
            "recorded": False,
            "error": "The decision could not be saved. Please try again.",
        }


def run_agent(
    question: str,
    session_id: str | None = None,
    customer_id: str | None = None,
) -> dict:
    """Answer one chat question using checkpointed conversation memory."""
    session_id = session_id or str(uuid4())
    try:
        logger.info(
            "agent_question=%r session_id=%s customer_id=%s",
            question,
            session_id,
            customer_id,
        )
        final_state = get_workflow().invoke(
            {"messages": [HumanMessage(question)]},
            config=_thread_config(session_id),
            context=Context(customer_id=customer_id),
        )
        return {
            "answer": _last_ai_text(final_state["messages"]),
            "session_id": session_id,
            "error": None,
        }
    except Exception:
        logger.exception("agent_failed session_id=%s", session_id)
        return {
            "answer": None,
            "session_id": session_id,
            "error": "Something went wrong. Please try again.",
        }


def stream_agent(
    question: str,
    session_id: str | None = None,
    customer_id: str | None = None,
):
    """Yield assistant text chunks as the agent produces them."""
    session_id = session_id or str(uuid4())
    try:
        for _namespace, payload in get_workflow().stream(
            {"messages": [HumanMessage(question)]},
            config=_thread_config(session_id),
            context=Context(customer_id=customer_id),
            stream_mode="messages",
            subgraphs=True,
        ):
            chunk = payload[0]
            text = getattr(chunk, "content", "")
            if text and isinstance(chunk, (AIMessage, AIMessageChunk)):
                yield _normalise_citations(text)
    except Exception:
        logger.exception("agent_stream_failed session_id=%s", session_id)
        yield "Something went wrong. Please try again."


def reset_session(session_id: str) -> None:
    """Forget one conversation thread's checkpointed history."""
    get_checkpointer().delete_thread(session_id)
