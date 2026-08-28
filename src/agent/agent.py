"""Layer 1 — the tool-calling agent, built by LangChain's create_agent.

create_agent owns the whole reason/act loop, parallel tool execution, and
session memory. Nothing in this file hand-rolls any of that.

Context.customer_id is the fix for the old "please give me your policy ID"
bug: it is passed at invoke time and injected straight into the customer-scoped
tools via ToolRuntime, so the model never has to ask for an ID it already has.
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver

from config import settings
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Caps one run so a confused loop cannot burn the Groq daily quota. "end"
# means it returns what it has instead of raising.
#
# Raised from 5 to 10: a single question answered in five calls, but a claim
# recommendation gathers evidence with several tools and then writes a
# four-part answer. At five the agent ran out mid-run and the employee saw
# "Model call limits exceeded: run limit (5/5)" where the reasoning should
# have been -- intermittently, since it sat right on the boundary.
MAX_MODEL_CALLS_PER_RUN = 10

# Shared by both prompts below so the format only has to change in one
# place -- it also has to match workflow.py's WRITE_RECOMMENDATION_PROMPT and
# recommendation.py's ClaimExplanation field description, which state it
# again for their own reasons (a per-invocation message and a schema
# description can't import a system-prompt constant), but at least the two
# agents built in this file share one source for it.
CITATION_FORMAT = "[Source: {insurer}, UIN: {uin}, Page {page}]"

# The chat agent: full tool access, free-text answers to whatever an
# employee asks about a customer's policies, claims and cover. It never
# assesses a claim -- it has no eligibility-engine tool to do that with, and
# is told not to try.
CHAT_SYSTEM_PROMPT = f"""You are a claims analysis copilot for an insurance claims employee.

WHO YOU ARE TALKING TO: a trained claims professional inside the insurance
company, not the customer. Answer their questions about a customer's
policies, claims and cover.

Rules:
- You only help with this customer's insurance policies, claims and cover.
  If a question is not about that -- general knowledge, current events,
  anything unrelated to insurance -- do not answer it, even if you know the
  answer. Say exactly: "I can only help with this customer's insurance
  policies and claims." Do not soften this into a partial answer.
- Your facts come from two kinds of source, and each has its own rule.
  1. The customer's own records -- profile, policies, claims, and any amount
     calculated from them. These are authoritative. State them directly with
     no citation. A record you were handed is never "not found".
  2. Policy document text retrieved for this question. State only what is
     explicitly written there. Never fill a gap with general insurance
     knowledge from your own training.
- If the answer needs policy document wording and the retrieved text does
  not contain it, say exactly:
  "I could not find this in the policy documents."
- For every fact taken from policy document text, end the sentence with a
  citation in this exact format:
  {CITATION_FORMAT}
  Use plain ASCII square brackets [ ] only -- never full-width brackets or
  any other marker. Never state a rule or condition quoted from a policy
  document without this citation immediately after it. If a retrieved chunk
  has no UIN or page, do not state the fact it carries. This citation rule
  applies to policy document text only -- never withhold a fact from the
  customer's records because it has no page number.
- The tools are named exactly: get_customer, get_policies, get_claims,
  check_coverage, check_exclusion, check_waiting_period,
  waiting_period_tracker, sum_insured_balance, calculate_payable_amount.
  Call them by these exact names -- do not add a "get_" prefix to any name
  that does not already have one.
- Never guess a fact about a customer, a policy, a date, or a rupee amount.
  Always call a tool to get it.
- You already know which customer this conversation is about -- never ask
  for a customer ID or a policy ID. Call get_policies to look up their
  policies, then use the policy_id and UIN it returns.
- Never do arithmetic yourself -- call calculate_payable_amount,
  sum_insured_balance, or waiting_period_tracker instead.
- For any waiting-period question, use exactly three steps: get_policies for
  the policy_id and UIN, then check_waiting_period once for the clause to
  cite, then waiting_period_tracker for the exact date. The clause often says
  "as specified below" without a number -- that is expected. Do NOT search
  again looking for the number; the tracker supplies it. Never run the same
  search twice with reworded queries.
- If a tool returns an error, say plainly what went wrong -- never invent an
  answer to cover for it.
- You do not assess or decide claims in this conversation -- you have no
  tool for that. If asked whether a claim should be approved, say that goes
  through the claim review flow, not chat."""

# The claim-explanation agent: no tools, response_format=ClaimExplanation
# (see get_claims_agent). The deterministic engine has already produced the
# status and, where applicable, the payable amount -- this agent's only job
# is to explain that finished result in the employee's language. It never
# gathers evidence itself, which is why it needs no tools at all.
CLAIM_EXPLANATION_SYSTEM_PROMPT = f"""You are a claims analysis copilot for an insurance claims employee.

WHO YOU ARE TALKING TO: a trained claims professional inside the insurance
company, reviewing a claim. You are not talking to the customer. The employee
makes the final call -- you write the explanation they will read before they
accept, edit or overturn it.

WHAT YOU RECEIVE: a deterministic eligibility engine has already run the full
checklist -- coverage, exclusion, waiting period, sub-limit, co-pay,
deductible, remaining sum insured -- and reached a status and, where
applicable, a payable amount. Both are handed to you as fact in the message
below. Your only job is to explain that result in plain language. You have no
tools in this role and are not asked to use any.

NEVER PHRASE ANYTHING AS FINAL OR BINDING. You do not approve or reject
claims -- the engine's status is a recommendation for the employee to act on,
not a decision. Write "Recommend: Approve" or "Recommend: Reject", never
"your claim is approved" or "we have rejected this claim".

Rules:
- Never recalculate the payable amount, never round it, and never state a
  figure you were not given. The number in the engine result is final --
  your job is to explain how it was reached, not to check it.
- Never change the status the engine reached. If it says needs_more_info,
  your recommendation is "Needs More Info", never a guess at approve or
  reject.
- Never invent evidence. State only the policy clauses and facts you were
  actually given -- a citation you were not handed does not exist for this
  answer.
- Read each fact's status before writing about it. "unknown" means the fact
  was never established -- report it under Missing information, never as
  zero or as not applying. "not_applicable" is the opposite: the policy was
  checked and confirmed to state no such condition -- report it as a
  finding, in the same words as its detail, not as something absent or
  unlocatable.
- For every policy-derived fact, keep its citation in this exact format:
  {CITATION_FORMAT}
  Use plain ASCII square brackets [ ] only. Never state a policy fact
  without the citation it came with, and never add a citation you were not
  given.
- Always name what is missing when the status is needs_more_info, or state
  plainly that nothing is missing.
- This is a recommendation for a human to review, never a final answer."""


@dataclass
class Context:
    """Per-request context. customer_id reaches tools via ToolRuntime.context."""

    customer_id: str | None = None


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """Return the one thread store, so memory survives a restart.

    Only the outer workflow passes this to compile(). The agent below is
    added to that workflow as a subgraph node, and LangGraph gives a subgraph
    its parent's checkpointer automatically -- handing the agent its own as
    well meant two objects writing the same threads, which is the kind of
    duplicate persistence that makes a resumed interrupt hard to reason about.

    check_same_thread=False because FastAPI serves requests from a threadpool.

    The explicit serde: get_claims_agent()'s response_format=ClaimExplanation
    means review_node's state -- checkpointed on every interrupt() so a
    resume can rebuild it -- carries that Pydantic type. LangGraph's default
    serializer allows any custom type through with a warning ("this will be
    blocked in a future version"), because loading an unrecognised type from
    an untrusted checkpoint file is a code-execution risk it does not want to
    take silently. Registering ClaimExplanation by name is the supported way
    to say this one is expected, rather than leaving every checkpoint load on
    the permissive-with-a-warning path.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    from src.agent.recommendation import ClaimExplanation

    checkpoint_path = settings.root_dir / "data" / "checkpoints.db"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[ClaimExplanation])
    return SqliteSaver(connection, serde=serde)


def _build_agent(system_prompt, tools=ALL_TOOLS, response_format=None):
    """Build one compiled agent. Shared by the chat and claim variants below.

    system_prompt, tools and response_format are the only things that differ
    between them -- model and middleware are identical either way.
    """
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for the agent")

    agent = create_agent(
        model=ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            # streaming=True makes the model emit token callbacks even when
            # create_agent calls invoke(), which is what lets
            # stream_mode="messages" paint the answer word by word.
            streaming=True,
        ),
        tools=tools,
        system_prompt=system_prompt,
        context_schema=Context,
        response_format=response_format,
        # No checkpointer here on purpose -- see get_checkpointer().
        middleware=[
            # Groq validates tool names server-side and raises an APIError for
            # the whole request if the model invents one (it has guessed
            # "get_sum_insured_balance"). One retry recovers the turn instead
            # of losing the employee's question.
            ModelRetryMiddleware(max_retries=2, on_failure="continue"),
            ModelCallLimitMiddleware(
                run_limit=MAX_MODEL_CALLS_PER_RUN,
                exit_behavior="end",
            ),
        ],
    )
    logger.info(
        "agent_built tools=%s model=%s structured=%s",
        len(tools or []), settings.llm_model, response_format is not None,
    )
    return agent


@lru_cache(maxsize=1)
def get_agent():
    """The chat agent: free-text replies, for /chat and run_agent().

        User -> Chat Agent -> Tools -> Answer

    CHAT_SYSTEM_PROMPT, full tool access, no response_format -- a chat
    answer is prose, not one fixed shape, and stream_agent() reads
    token-by-token AIMessage chunks that structured output does not produce
    the same way.
    """
    return _build_agent(CHAT_SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def get_claims_agent():
    """The claim-explanation agent: structured replies, for the review node.

        Claim -> Eligibility Engine -> Claim Explanation Agent -> Human Review

    CLAIM_EXPLANATION_SYSTEM_PROMPT, no tools, response_format=ClaimExplanation.
    Two differences from get_agent(), and they are linked, not incidental:

    response_format=ClaimExplanation makes the model's reply validated JSON
    instead of free text, which review_node reads directly from
    state["structured_response"].reasoning -- no scanning the message list
    for the last AIMessage. status and payable_amount are not part of the
    schema the model fills in; see ClaimExplanation for why.

    tools=None: Groq's API rejects a request that combines its JSON
    response-format mode with function/tool calling in the same call --
    "json mode cannot be combined with tool/function calling". This node
    does not lose anything by going tool-less: WRITE_RECOMMENDATION_PROMPT
    already hands it the finished eligibility dict, evidence and all, so its
    only job is turning that dict into readable prose. It was never the node
    that looks anything up.
    """
    from src.agent.recommendation import ClaimExplanation

    return _build_agent(
        CLAIM_EXPLANATION_SYSTEM_PROMPT, tools=None, response_format=ClaimExplanation
    )
