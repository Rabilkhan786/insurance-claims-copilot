"""LangChain agents for employee chat and claim explanations."""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, ModelRetryMiddleware
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver

from config import settings
from src.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

MAX_MODEL_CALLS_PER_RUN = 10
CITATION_FORMAT = "[Source: {insurer}, UIN: {uin}, Page {page}]"

CHAT_SYSTEM_PROMPT = f"""You are a claims analysis copilot for an insurance claims employee.

You help with this customer's policies, claims, and cover. Do not answer unrelated questions. If a question is unrelated, say: "I can only help with this customer's insurance policies and claims."

Use the customer's SQL records for customer, policy, claim, and calculated-amount facts. Use retrieved policy document text for policy wording. Do not fill missing policy information from general knowledge.

For policy-document facts, use the exact citation format {CITATION_FORMAT}. Use ASCII square brackets only. Do not state an uncited policy-document fact. If retrieved wording lacks a UIN or page, do not use that wording as evidence.

Available tools: get_customer, get_policies, get_claims, check_coverage, check_exclusion, check_waiting_period, waiting_period_tracker, sum_insured_balance, calculate_payable_amount.

Never guess customer, policy, date, or money values. Use the appropriate tool. Do not ask for a customer or policy ID already available in context.

For waiting-period questions: get_policies, check_waiting_period once for cited wording, then waiting_period_tracker. Do not repeat the same retrieval with reworded queries.

If a tool fails, report the failure plainly. Do not make claim approval decisions in chat; those belong to the claim review flow."""

CLAIM_EXPLANATION_SYSTEM_PROMPT = f"""You are a claims analysis copilot for an insurance claims employee reviewing a claim.

A deterministic eligibility engine has already checked coverage, exclusions, waiting period, sub-limit, co-pay, deductible, and remaining sum insured. Your job is only to explain that result clearly.

The employee makes the final decision. Write "Recommend: Approve", "Recommend: Reject", or "Recommend: Needs More Info". Never present the result as a final or binding decision.

Do not recalculate, round, change, or invent amounts. Do not invent evidence. Preserve the policy citations you are given in this exact format: {CITATION_FORMAT}.

Use fact status correctly: "unknown" belongs in missing information; "not_applicable" means the policy was checked and the condition does not apply.

Always identify missing information when the result is needs_more_info."""


@dataclass
class Context:
    """Per-request context passed to customer-scoped tools."""

    customer_id: str | None = None


@lru_cache(maxsize=1)
def get_checkpointer() -> SqliteSaver:
    """Return the shared SQLite checkpointer."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from src.agent.recommendation import ClaimExplanation

    checkpoint_path = settings.root_dir / "data" / "checkpoints.db"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[ClaimExplanation])
    return SqliteSaver(connection, serde=serde)


def _build_agent(system_prompt, tools=ALL_TOOLS, response_format=None):
    """Build an agent with shared model and middleware settings."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required for the agent")

    agent = create_agent(
        model=ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            streaming=True,
        ),
        tools=tools,
        system_prompt=system_prompt,
        context_schema=Context,
        response_format=response_format,
        middleware=[
            ModelRetryMiddleware(max_retries=2, on_failure="continue"),
            ModelCallLimitMiddleware(
                run_limit=MAX_MODEL_CALLS_PER_RUN,
                exit_behavior="end",
            ),
        ],
    )
    logger.info(
        "agent_built tools=%s model=%s structured=%s",
        len(tools or []),
        settings.llm_model,
        response_format is not None,
    )
    return agent


@lru_cache(maxsize=1)
def get_agent():
    """Return the chat agent."""
    return _build_agent(CHAT_SYSTEM_PROMPT)


@lru_cache(maxsize=1)
def get_claims_agent():
    """Return the structured claim-explanation agent."""
    from src.agent.recommendation import ClaimExplanation

    return _build_agent(
        CLAIM_EXPLANATION_SYSTEM_PROMPT,
        tools=None,
        response_format=ClaimExplanation,
    )
