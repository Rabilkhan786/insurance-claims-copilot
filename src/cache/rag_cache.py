"""In-process TTL cache for hybrid retrieval results.

WHY: a full retrieval is dense search + sparse search + RRF + cross-encoder
rerank, which costs roughly 4 seconds. Demo users ask the same handful of
questions repeatedly, and the eligibility engine re-queries the same treatment
for coverage and exclusions, so the same (query, topic, uin) triple comes round
often. A hit returns in microseconds.

The cache key is a hash of query + topic + uin, so a question filtered to one
policy never returns another policy's chunks.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time

from config import settings

logger = logging.getLogger(__name__)

# key -> (expires_at_epoch_seconds, hits)
_entries: dict[str, tuple[float, list[dict]]] = {}


def _make_key(
    query: str,
    topic: str | list[str] | None,
    uin: str | None,
    extra: dict | None = None,
) -> str:
    """Hash everything that changes what retrieval returns."""
    payload = json.dumps(
        {
            "query": (query or "").strip().lower(),
            "topic": sorted(topic) if isinstance(topic, list) else topic,
            "uin": uin,
            "extra": extra,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(
    query: str,
    topic: str | list[str] | None = None,
    uin: str | None = None,
    extra: dict | None = None,
) -> list[dict] | None:
    """Return cached hits, or None if absent or expired."""
    key = _make_key(query, topic, uin, extra)
    entry = _entries.get(key)
    if entry is None:
        return None

    expires_at, hits = entry
    if time.time() >= expires_at:
        # Expired -- drop it so the dict does not grow forever.
        _entries.pop(key, None)
        return None

    print(f"rag_cache: HIT for {query[:50]!r}")
    return hits


def set(
    query: str,
    hits: list[dict],
    topic: str | list[str] | None = None,
    uin: str | None = None,
    extra: dict | None = None,
) -> None:
    """Store hits under the query/topic/uin key for the configured TTL."""
    key = _make_key(query, topic, uin, extra)
    _entries[key] = (time.time() + settings.rag_cache_ttl_seconds, hits)
    logger.info("rag_cache_store query=%r entries=%s", query[:60], len(_entries))
