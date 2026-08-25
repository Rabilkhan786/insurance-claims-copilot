"""Agent-facing tools over the hybrid RAG pipeline.

Every function narrows retrieval with a Pinecone metadata filter (topic
and/or policy UIN) instead of searching the whole index — a question about
waiting periods should only compete against waiting_period chunks.

The filter matches on `topics`, the list of every topic a chunk has real
evidence for, rather than on the single `topic` label. A clause like
"cataract is covered up to Rs 40,000" is genuinely both coverage and a
sub-limit, and filtering on one label made it invisible to the other tool.
"""
from __future__ import annotations

from langchain_core.tools import tool

from src.cache import rag_cache
from src.retrieval import retrieve, warmup as _warmup_retrieval


def _build_filter(
    topic: str | list[str] | None,
    policy_uin: str | None,
) -> dict | None:
    """Combine an optional topic (or list of topics) and UIN into one filter.

    Pinecone's $in matches when any element of an array field is in the given
    list, so a chunk tagged ["coverage", "sub_limit"] is found by a search for
    either one.
    """
    metadata_filter = {}
    if isinstance(topic, list):
        metadata_filter["topics"] = {"$in": topic}
    elif topic:
        metadata_filter["topics"] = {"$in": [topic]}
    if policy_uin:
        metadata_filter["uin"] = policy_uin
    return metadata_filter or None


def _format_hits(documents: list) -> list[dict]:
    """Trim retrieved Documents down to what the LLM needs to answer and cite.

    Anything without a UIN and page cannot be cited, and the system prompt
    forbids stating an uncitable fact -- but they are still returned so the
    model can see that something was found and say the evidence was thin.
    """
    formatted = []
    for document in documents:
        metadata = document.metadata or {}
        formatted.append(
            {
                "text": document.page_content,
                "uin": metadata.get("uin"),
                "insurer": metadata.get("insurer"),
                "page": metadata.get("page"),
                "topic": metadata.get("topic"),
            }
        )
    return formatted


def _search(
    query: str,
    topic: str | list[str] | None = None,
    policy_uin: str | None = None,
    extra_filter: dict | None = None,
) -> list[dict]:
    """Run one cached hybrid retrieval and format the hits for the LLM.

    Every RAG tool goes through here so they all share one cache and one
    filter-building path.
    """
    cached = rag_cache.get(query, topic=topic, uin=policy_uin, extra=extra_filter)
    if cached is not None:
        return cached

    metadata_filter = _build_filter(topic, policy_uin)
    if extra_filter:
        metadata_filter = {**(metadata_filter or {}), **extra_filter}

    hits = _format_hits(retrieve(query, metadata_filter))
    rag_cache.set(query, hits, topic=topic, uin=policy_uin, extra=extra_filter)
    return hits


def warmup() -> None:
    """Load the embedder and reranker before the first real request."""
    _warmup_retrieval()


@tool
def check_coverage(treatment: str, policy_uin: str | None = None) -> list[dict]:
    """Find policy clauses that state whether a treatment or procedure is
    covered. Also searches sub-limit chunks (e.g. 'cataract covered up to
    Rs 40,000'). Pass policy_uin to narrow to one plan. Returns clauses with
    UIN, page, and insurer for citation."""
    return _search(treatment, topic=["coverage", "sub_limit"], policy_uin=policy_uin)


@tool
def check_exclusion(treatment: str, policy_uin: str | None = None) -> list[dict]:
    """Find policy clauses that state whether a treatment or condition is
    excluded from coverage. Pass policy_uin to narrow to one plan. Returns
    clauses with UIN, page, and insurer for citation."""
    return _search(treatment, topic="exclusion", policy_uin=policy_uin)


@tool
def check_waiting_period(condition: str, policy_uin: str | None = None) -> list[dict]:
    """Find policy clauses that state the waiting period for a condition
    (e.g. maternity, pre-existing disease). Returns clause text with UIN and
    page. To compute whether a customer is currently eligible, also call
    waiting_period_tracker to turn that into an exact eligibility date."""
    return _search(condition, topic="waiting_period", policy_uin=policy_uin)
