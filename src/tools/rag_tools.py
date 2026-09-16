"""LangChain tools for policy-document retrieval."""
from __future__ import annotations

from langchain_core.tools import tool

from src.cache import rag_cache
from src.retrieval import retrieve, warmup as _warmup_retrieval

COVERAGE_TOPICS = ["coverage", "sub_limit", "copay"]


def _build_filter(
    topic: str | list[str] | None,
    policy_uin: str | None,
) -> dict | None:
    """Build a Pinecone metadata filter from topic and policy UIN."""
    metadata_filter = {}

    if isinstance(topic, list):
        metadata_filter["topics"] = {"$in": topic}
    elif topic:
        metadata_filter["topics"] = {"$in": [topic]}

    if policy_uin:
        metadata_filter["uin"] = policy_uin

    return metadata_filter or None


def _format_hits(documents: list) -> list[dict]:
    """Return only the document fields needed by the agent."""
    hits = []
    for document in documents:
        metadata = document.metadata or {}
        hits.append(
            {
                "text": document.page_content,
                "uin": metadata.get("uin"),
                "insurer": metadata.get("insurer"),
                "page": metadata.get("page"),
                "topic": metadata.get("topic"),
            }
        )
    return hits


def _search(
    query: str,
    topic: str | list[str] | None = None,
    policy_uin: str | None = None,
    extra_filter: dict | None = None,
) -> list[dict]:
    """Run cached hybrid retrieval and return formatted hits."""
    cached = rag_cache.get(
        query,
        topic=topic,
        uin=policy_uin,
        extra=extra_filter,
    )
    if cached is not None:
        return cached

    metadata_filter = _build_filter(topic, policy_uin)
    if extra_filter:
        metadata_filter = {**(metadata_filter or {}), **extra_filter}

    hits = _format_hits(retrieve(query, metadata_filter))
    rag_cache.set(
        query,
        hits,
        topic=topic,
        uin=policy_uin,
        extra=extra_filter,
    )
    return hits


def warmup() -> None:
    """Load retrieval models before the first request."""
    _warmup_retrieval()


@tool
def check_coverage(
    treatment: str,
    policy_uin: str | None = None,
) -> list[dict]:
    """Find coverage, sub-limit, and co-pay clauses for a treatment."""
    return _search(
        treatment,
        topic=COVERAGE_TOPICS,
        policy_uin=policy_uin,
    )


@tool
def check_exclusion(
    treatment: str,
    policy_uin: str | None = None,
) -> list[dict]:
    """Find exclusion clauses for a treatment or condition."""
    return _search(
        treatment,
        topic="exclusion",
        policy_uin=policy_uin,
    )


@tool
def check_waiting_period(
    condition: str,
    policy_uin: str | None = None,
) -> list[dict]:
    """Find waiting-period clauses for a condition."""
    return _search(
        condition,
        topic="waiting_period",
        policy_uin=policy_uin,
    )
