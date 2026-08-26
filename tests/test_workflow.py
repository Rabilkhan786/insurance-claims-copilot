"""Tests for the human-in-the-loop half of the claim workflow.

The graph is exercised with a stub agent node and a stub engine, so these run
without Groq or Pinecone. What is being tested is the LangGraph wiring --
that the graph really stops at review, that resuming the same thread_id picks
it back up, and that one audit row lands holding both sides of the decision.
"""
from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command

from src.agent import workflow as workflow_module
from src.agent.recommendation import ClaimRecommendation
from src.decisions import DecisionStore
from src.eligibility import ELIGIBLE


class _State(MessagesState, total=False):
    claim: dict | None
    claim_id: str | None
    policy_id: str | None
    customer_id: str | None
    eligibility: dict | None
    recommendation: dict | None
    decision: dict | None
    error: str | None


ENGINE_RESULT = {
    "status": ELIGIBLE,
    "bill_amount": 60000,
    "covered_amount": 40000,
    "copay_amount": 2000,
    "estimated_payable": 38000,
    "reason": "Cataract is capped at Rs 40,000, then a 5% co-pay applies.",
    "evidence": [],
    "missing_information": [],
    "facts": {},
    "deductions": {},
}


@pytest.fixture
def graph(tmp_path, monkeypatch):
    """The real review and persist nodes, wired to a throwaway store.

    Only the eligibility and agent nodes are stubbed -- those are the two that
    need the network. review_node and persist_decision_node are the code under
    test and run exactly as they do in production.
    """
    store = DecisionStore(tmp_path / "crm.db")
    monkeypatch.setattr(workflow_module, "get_decision_store", lambda: store)

    def eligibility_node(state):
        return {"eligibility": ENGINE_RESULT, "customer_id": "CUST001"}

    def agent_node(state):
        from langchain_core.messages import AIMessage

        return {"messages": [AIMessage("Recommend: approve. Payable Rs 38,000.")]}

    builder = StateGraph(_State)
    builder.add_node("eligibility", eligibility_node)
    builder.add_node("agent", agent_node)
    builder.add_node("review", workflow_module.review_node)
    builder.add_node("persist_decision", workflow_module.persist_decision_node)
    builder.add_edge(START, "eligibility")
    builder.add_edge("eligibility", "agent")
    builder.add_edge("agent", "review")
    builder.add_edge("review", "persist_decision")
    builder.add_edge("persist_decision", END)

    return builder.compile(checkpointer=InMemorySaver()), store


def _config(thread_id: str = "t-1") -> dict:
    return {"configurable": {"thread_id": thread_id}}


# --- the pause ---------------------------------------------------------------
def test_the_graph_stops_at_review_and_hands_back_the_recommendation(graph):
    compiled, _ = graph

    state = compiled.invoke(
        {"claim": {"treatment": "Cataract Surgery"}, "claim_id": "CLM-1"},
        config=_config(),
    )

    interrupts = state.get("__interrupt__")
    assert interrupts, "the graph ran to completion instead of pausing for review"

    payload = dict(interrupts[0].value)
    assert payload["recommendation"]["status"] == "approve"
    assert payload["recommendation"]["payable_amount"] == 38000


def test_nothing_is_recorded_until_the_employee_answers(graph):
    compiled, store = graph

    compiled.invoke({"claim": {}, "claim_id": "CLM-1"}, config=_config())

    assert store.recent() == []


# --- the resume --------------------------------------------------------------
def test_resuming_the_same_thread_records_the_decision(graph):
    compiled, store = graph
    config = _config("t-resume")
    compiled.invoke({"claim": {}, "claim_id": "CLM-2"}, config=config)

    compiled.invoke(
        Command(resume={"decision": "approve", "decided_by": "emp.demo"}),
        config=config,
    )

    rows = store.recent()
    assert len(rows) == 1
    assert rows[0]["claim_id"] == "CLM-2"
    assert rows[0]["employee_decision"] == "approve"
    assert rows[0]["ai_decision"] == "approve"


def test_an_edited_amount_leaves_the_recommendation_untouched(graph):
    """The audit's whole value is that both figures survive."""
    compiled, store = graph
    config = _config("t-edit")
    compiled.invoke({"claim": {}, "claim_id": "CLM-3"}, config=config)

    compiled.invoke(
        Command(
            resume={
                "decision": "edit",
                "decided_by": "emp.demo",
                "payable_amount": 25000,
                "edits": "Second eye billed separately.",
            }
        ),
        config=config,
    )

    row = store.recent()[0]
    assert row["ai_payable_amount"] == 38000
    assert row["employee_payable_amount"] == 25000
    assert row["ai_recommendation"]["payable_amount"] == 38000


def test_two_claims_on_two_threads_do_not_collide(graph):
    """Resuming one thread must not finish the other's review."""
    compiled, store = graph
    first, second = _config("t-a"), _config("t-b")
    compiled.invoke({"claim": {}, "claim_id": "CLM-A"}, config=first)
    compiled.invoke({"claim": {}, "claim_id": "CLM-B"}, config=second)

    compiled.invoke(
        Command(resume={"decision": "reject", "decided_by": "emp.demo",
                        "reason": "Duplicate claim."}),
        config=second,
    )

    rows = store.recent()
    assert [row["claim_id"] for row in rows] == ["CLM-B"]


# --- idempotency -------------------------------------------------------------
def test_review_node_builds_the_same_recommendation_on_both_passes():
    """Everything before interrupt() runs twice, so it has to be pure.

    review_node builds the recommendation, then calls interrupt(). On resume
    the node re-enters from the top and builds it again -- if that were not a
    pure function of state, the employee would approve one thing and the audit
    would record another.
    """
    from langchain_core.messages import AIMessage

    state = {
        "eligibility": ENGINE_RESULT,
        "messages": [AIMessage("Recommend: approve.")],
    }

    first = ClaimRecommendation.from_engine(
        state["eligibility"], reasoning=workflow_module._last_ai_text(state["messages"])
    )
    second = ClaimRecommendation.from_engine(
        state["eligibility"], reasoning=workflow_module._last_ai_text(state["messages"])
    )

    assert first.model_dump() == second.model_dump()


def test_citations_are_normalised_to_ascii_brackets():
    """The model sometimes emits full-width brackets despite the prompt."""
    normalised = workflow_module._normalise_citations(
        "Covered 【Source: Star Health, UIN: X, Page 8】"
    )

    assert normalised == "Covered [Source: Star Health, UIN: X, Page 8]"
