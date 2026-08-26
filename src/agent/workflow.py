"""Layer 2 — the outer workflow: a claim goes in, a reviewed decision comes out.

    claim submitted -> eligibility -> agent -> review (pauses) -> persist -> END
    chat question   -> agent -> END

The compiled create_agent graph from Layer 1 is added here as an ordinary node
(subgraph-as-node), so the claim path ends with the agent turning a decision
the deterministic engine already made into readable reasoning for an employee.

The review step uses LangGraph's interrupt(): the graph stops, its state is
checkpointed, and the caller is handed the recommendation. Nothing here
hand-rolls a "waiting for approval" flag -- the client resumes the same
thread_id with Command(resume=...) and the graph picks up where it stopped.

customer_id is never parsed out of message text -- it arrives as Context at
invoke time and reaches the tools through ToolRuntime.
"""
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
    """MessagesState supplies `messages`; these are the claim-path extras."""

    claim: dict[str, Any] | None
    claim_id: str | None
    policy_id: str | None
    # Persisted in state, not read from Context at the end: the graph is
    # resumed with Command(resume=...), which carries no Context, so
    # runtime.context is None in every node after the interrupt.
    customer_id: str | None
    eligibility: dict[str, Any] | None
    # Written by explain_claim (the structured-output agent), read once by
    # review_node and never touched again. ClaimExplanation, not
    # ClaimRecommendation -- see recommendation.py for why the two are
    # different shapes.
    structured_response: Any | None
    # The typed ClaimRecommendation, flattened. Built once in review_node so
    # the payload the employee saw is exactly what the audit row stores.
    recommendation: dict[str, Any] | None
    decision: dict[str, Any] | None
    error: str | None


# The engine has already decided. The agent's only job here is to explain that
# decision to an employee -- which is why the figures are declared final.
WRITE_RECOMMENDATION_PROMPT = """Write a claim recommendation for the claims \
employee reviewing this file. The figures below are final -- do not \
recalculate them, do not round them, and do not introduce any amount that is \
not listed.

Structure it as:
1. Recommend: approve / reject / needs more info -- and why, in one sentence.
2. The payable amount, and which deduction reduced the bill to it. If the \
status is needs_more_info there is no payable amount yet -- say what has to \
be established before one can be calculated, and do not invent a figure.
3. The policy clauses that support the decision, each with its citation in \
the exact format [Source: {{insurer}}, UIN: {{uin}}, Page {{page}}].
4. Anything missing that the employee should chase before deciding.

Read the per-fact statuses before you write. Only "unknown" belongs under \
"Missing information" or as something you "could not find" -- that status \
means the fact was never established. A "not_applicable" fact is the \
opposite: it means the policy was checked and confirmed to state no such \
condition. Report it as a finding in section 3, in the same words as its \
detail (e.g. "this plan carries no deductible; it pays from rupee one"), \
never as something absent or unlocatable.

This is a recommendation for a human to review, never a final answer. Do not \
write "the claim is approved" -- write "recommend: approve".

Engine result:
{eligibility}

Claim as submitted:
{claim}"""


@lru_cache(maxsize=1)
def get_decision_store() -> DecisionStore:
    """Return the shared audit-trail store, building it once per process."""
    return DecisionStore(settings.crm_db_path)


def _to_bill_data(claim: dict) -> dict:
    """Map the employee's form fields onto what the engine expects.

    check_eligibility reads `treatment`/`diagnosis` and `total_amount`. It has
    no idea the dict came from a form rather than anywhere else, which is why
    the engine itself needed no change when the bill-upload path was removed.
    """
    return {
        "treatment": claim.get("treatment") or claim.get("procedure") or "",
        "diagnosis": claim.get("diagnosis") or "",
        "total_amount": claim.get("claim_amount") or claim.get("total_amount") or 0,
        "hospital": claim.get("hospital") or "",
        "admission_date": claim.get("treatment_date") or claim.get("admission_date"),
    }


def eligibility_node(state: WorkflowState, runtime: Runtime[Context]) -> dict:
    """Run the deterministic checklist, then queue its result for explanation."""
    customer_id = runtime.context.customer_id
    policy_id = state.get("policy_id")
    if not customer_id or not policy_id:
        return {"error": "A customer ID and policy ID are both needed to check a claim."}

    claim = state.get("claim") or {}
    bill_data = _to_bill_data(claim)
    print(f"eligibility_node: checking {bill_data.get('treatment')!r} for {customer_id}")

    eligibility = check_eligibility(bill_data, customer_id, policy_id)
    print(
        f"eligibility_node: status={eligibility.get('status')} "
        f"payable={eligibility.get('estimated_payable')}"
    )

    return {
        "eligibility": eligibility,
        "customer_id": customer_id,
        "messages": [
            HumanMessage(
                WRITE_RECOMMENDATION_PROMPT.format(
                    eligibility=eligibility, claim=claim
                )
            )
        ],
    }


def _explanation_text(state: WorkflowState) -> str:
    """Read the model's prose, structured output first.

    explain_claim runs with response_format=ClaimExplanation, so its answer
    lands in structured_response.reasoning rather than as a plain AIMessage
    -- no scanning the message list for the last one. The messages-based
    fallback exists for the state a caller builds by hand (this file's own
    tests stub the agent node without response_format), not for anything the
    real graph produces.
    """
    structured = state.get("structured_response")
    if structured is not None:
        return _normalise_citations(structured.reasoning)
    return _last_ai_text(state.get("messages", []))


def review_node(state: WorkflowState) -> dict:
    """Hand the recommendation to the employee and wait for their call.

    interrupt() raises the first time through: LangGraph checkpoints the state
    and returns this payload to whoever invoked the graph. When the client
    resumes with Command(resume=...), the node runs again from the top and
    interrupt() returns that value instead of raising -- so everything above
    it has to be safe to run twice. It is: building the recommendation is a
    pure function of state, with no writes and no tool calls.
    """
    recommendation = ClaimRecommendation.from_engine(
        state.get("eligibility") or {},
        reasoning=_explanation_text(state),
    )
    print(f"review_node: pausing for employee review ({recommendation.status})")

    decision = interrupt(
        {
            "recommendation": recommendation.model_dump(),
            "claim": state.get("claim"),
        }
    )

    print(f"review_node: employee chose {decision.get('decision')!r}")
    return {"recommendation": recommendation.model_dump(), "decision": decision}


def persist_decision_node(state: WorkflowState) -> dict:
    """Write the AI recommendation and the employee's decision side by side."""
    decision = state.get("decision") or {}
    claim = state.get("claim") or {}

    try:
        get_decision_store().record(
            claim_id=state.get("claim_id") or claim.get("claim_id") or str(uuid4()),
            customer_id=state.get("customer_id") or "",
            policy_id=state.get("policy_id"),
            # The whole typed recommendation is the audit record: the status,
            # the figure, the reasoning the employee actually read, and the
            # evidence behind it. It is never overwritten by their answer.
            recommendation=state.get("recommendation") or {},
            employee_decision=decision.get("decision", "approve"),
            employee_payable_amount=decision.get("payable_amount"),
            employee_edits=decision.get("edits"),
            override_reason=decision.get("reason"),
            decided_by=decision.get("decided_by") or "unknown",
        )
    except Exception:
        # A failed audit write must be loud in the log, but it should not
        # throw away the decision the employee already made.
        logger.exception("audit_write_failed claim_id=%s", state.get("claim_id"))
        return {"error": "The decision was made but could not be written to the audit trail."}

    return {}


def _route_entry(state: WorkflowState) -> str:
    """Send a submitted claim down the eligibility path, chat to the agent."""
    return "eligibility" if state.get("claim") else "agent"


def _after_eligibility(state: WorkflowState) -> str:
    """Only ask for an explanation of a decision that was actually produced.

    explain_claim always leads to review once reached -- there is no longer
    a branch here for "explained, but not actually a claim", because the two
    tasks now run on two different nodes instead of one shared one.
    """
    return END if state.get("error") else "explain_claim"


_workflow = None


def get_workflow():
    """Return the compiled outer workflow, building it once per process."""
    global _workflow
    if _workflow is None:
        builder = StateGraph(WorkflowState, context_schema=Context)

        builder.add_node("eligibility", eligibility_node)
        # Two agent instances, not one: explain_claim runs with
        # response_format=ClaimExplanation (see get_claims_agent), so its
        # reply is validated JSON review_node reads directly. agent stays
        # free text -- chat is prose, and stream_agent() reads token-by-token
        # AIMessage chunks that structured output does not produce the same
        # way.
        builder.add_node("agent", get_agent())
        builder.add_node("explain_claim", get_claims_agent())
        builder.add_node("review", review_node)
        builder.add_node("persist_decision", persist_decision_node)

        builder.add_conditional_edges(
            START, _route_entry, {"eligibility": "eligibility", "agent": "agent"}
        )
        builder.add_conditional_edges(
            "eligibility", _after_eligibility, {"explain_claim": "explain_claim", END: END}
        )
        builder.add_edge("explain_claim", "review")
        builder.add_edge("agent", END)
        builder.add_edge("review", "persist_decision")
        builder.add_edge("persist_decision", END)

        # interrupt() only works when the graph can persist itself, so the
        # checkpointer is required here, not optional.
        _workflow = builder.compile(checkpointer=get_checkpointer())
        logger.info("workflow_built")
    return _workflow


def _normalise_citations(text: str) -> str:
    """Force citation markers to plain ASCII brackets.

    The model sometimes emits full-width 【Source: ...】 despite the prompt.
    Rewriting the bracket characters is purely cosmetic -- it never changes
    the insurer, UIN or page inside -- and it keeps the citation format the
    UI and tests rely on stable.
    """
    return text.replace("【", "[").replace("】", "]")


def _last_ai_text(messages: list) -> str:
    """Find the final assistant message's text content."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content:
            return _normalise_citations(message.content)
    return "I could not find this in your policy documents."


def _thread_config(session_id: str) -> dict:
    """Build the config that scopes checkpointed state to one review thread."""
    return {"configurable": {"thread_id": session_id}}


def _interrupt_payload(state: dict) -> dict | None:
    """Pull the recommendation out of a graph run that stopped at review."""
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
    """Analyse one claim and stop at the review step.

    Returns the recommendation for the employee to act on. The graph is left
    paused on this session_id -- call submit_decision() to finish it.
    """
    session_id = session_id or str(uuid4())
    try:
        logger.info(
            "claim_review session_id=%s customer_id=%s policy_id=%s",
            session_id, customer_id, policy_id,
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
            # No interrupt means the run stopped early -- usually a missing
            # policy, caught in eligibility_node.
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
    """Resume a paused review with the employee's decision, and record it."""
    try:
        logger.info(
            "decision_submitted session_id=%s decision=%s by=%s",
            session_id, decision, decided_by,
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
    """Answer one chat question. Memory comes from the checkpointer's thread."""
    session_id = session_id or str(uuid4())
    try:
        logger.info(
            "agent_question=%r session_id=%s customer_id=%s",
            question, session_id, customer_id,
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
    """Yield the answer token by token as the model produces it.

    stream_mode="messages" hands back each LLM chunk as it arrives, so the
    UI can paint words immediately. subgraphs=True is required because the
    agent runs as a subgraph inside this workflow -- without it the answer
    arrives as one block. Tool-call chunks carry no text and are skipped.
    """
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
