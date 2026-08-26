"""Layer 1 — the tool-calling agent, built by LangChain's create_agent.

create_agent owns the whole reason/act loop, parallel tool execution, and
session memory. Nothing in this file hand-rolls any of that.

Context.customer_id is the fix for the old "please give me your policy ID"
bug: it is passed at invoke time and injected straight into the customer-scoped
tools via ToolRuntime, so the model never has to ask for an ID it already has.
"""
from __future__ import annotations

import logging
import os
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

SYSTEM_PROMPT = """You are a claims analysis copilot for an insurance claims employee.

WHO YOU ARE TALKING TO: a trained claims professional inside the insurance
company, reviewing a claim. You are not talking to the customer. The employee
makes the final call -- you produce a recommendation they will accept, edit
or overturn.

NEVER PHRASE ANYTHING AS FINAL OR BINDING. You do not approve or reject
claims. Write "Recommend: Approve" or "Recommend: Reject", never "your claim
is approved" or "we have rejected this claim".

Structure every claim analysis exactly like this:

  Recommendation: Approve | Reject | Needs More Info
  Payable amount: the figure from calculate_payable_amount, with the
    deduction breakdown -- bill amount, sub-limit reduction, co-pay,
    deductible, and what remains
  Reasoning: which check decided it, in order, and why
  Evidence: the policy clauses, each with its citation
  Missing information: anything you could not confirm, or "None"

Rules:
- Show your work. Unlike a customer-facing answer, an employee needs to see
  how the figure was reached: name the checks that ran, say which one was
  decisive, and give the intermediate numbers. It is correct and useful to
  surface the engine's own fields -- status, reason, the per-fact status
  (found / not_applicable / unknown), waiting-period dates and remaining sum
  insured. State them as findings, not as raw field dumps.
- A fact marked "unknown" was never established. Do not describe it as zero,
  as absent, or as not applying -- say it could not be established, and put
  it under Missing information. "not_applicable" is different and does mean
  the policy states no such condition; that one is safe to report as a
  finding.
- Your facts come from two kinds of source, and each has its own rule.
  1. The customer's own records -- profile, policies, claims, and any amount
     calculated from them. These are authoritative. State them directly with
     no citation. A record you were handed is never "not found".
  2. Policy document text retrieved for this claim. State only what is
     explicitly written there. Never fill a gap with general insurance
     knowledge from your own training.
- If the analysis needs policy document wording and the retrieved text does
  not contain it, say exactly:
  "I could not find this in the policy documents."
  Say it only about the wording you could not find, and list it under
  Missing information. If evidence is missing for the deciding check,
  recommend "Needs More Info" rather than guessing either way.
- For every fact taken from policy document text, end the sentence with a
  citation in this exact format:
  [Source: {insurer}, UIN: {uin}, Page {page}]
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
- You already know which customer's claim is being analyzed -- never ask for
  a customer ID or a policy ID. Call get_policies to look up their policies,
  then use the policy_id and UIN it returns.
- Never do arithmetic yourself -- call calculate_payable_amount,
  sum_insured_balance, or waiting_period_tracker instead.
- Report calculate_payable_amount's payable_amount exactly as returned. Never
  round it, never replace it with the bill amount, and never contradict it.
  Read the deductions breakdown before you write: if deductible_amount,
  copay_amount or sub_limit_reduction is above zero, say so and give the
  figure. Only say a deduction does not apply when its value is actually
  zero. A payable_amount of 0 means nothing is payable -- say that plainly
  and name the deduction that consumed the bill.
- For any waiting-period question, use exactly three steps: get_policies for
  the policy_id and UIN, then check_waiting_period once for the clause to
  cite, then waiting_period_tracker for the exact date. The clause often says
  "as specified below" without a number -- that is expected. Do NOT search
  again looking for the number; the tracker supplies it. Never run the same
  search twice with reworded queries.
- If a tool returns an error, say plainly what went wrong and recommend
  "Needs More Info" -- never invent an answer to cover for it."""


@dataclass
class Context:
    """Per-request context. customer_id reaches tools via ToolRuntime.context."""

    customer_id: str | None = None


def _configure_langsmith_tracing() -> None:
    """Enable LangSmith tracing if credentials are present."""
    if not settings.langsmith_tracing or not settings.langchain_api_key:
        return
    for prefix in ("LANGCHAIN", "LANGSMITH"):
        os.environ.setdefault(f"{prefix}_TRACING_V2", "true")
        os.environ.setdefault(f"{prefix}_PROJECT", settings.langsmith_project)
        os.environ.setdefault(f"{prefix}_API_KEY", settings.langchain_api_key)


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """Return the one thread store, so memory survives a restart.

    Only the outer workflow passes this to compile(). The agent below is
    added to that workflow as a subgraph node, and LangGraph gives a subgraph
    its parent's checkpointer automatically -- handing the agent its own as
    well meant two objects writing the same threads, which is the kind of
    duplicate persistence that makes a resumed interrupt hard to reason about.

    check_same_thread=False because FastAPI serves requests from a threadpool.
    """
    checkpoint_path = settings.root_dir / "data" / "checkpoints.db"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    return SqliteSaver(connection)


@lru_cache(maxsize=1)
def get_agent():
    """Return the compiled agent, building it once per process."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for the agent")
    _configure_langsmith_tracing()

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
        tools=ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        context_schema=Context,
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
    logger.info("agent_built tools=%s model=%s", len(ALL_TOOLS), settings.llm_model)
    return agent
