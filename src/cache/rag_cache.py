"""In-process TTL cache for hybrid retrieval results."""
from __future__ import annotations

import hashlib
import json
import logging
import time

from config import settings

logger = logging.getLogger(__name__)

_entries: dict[str, tuple[float, list[dict]]] = {}


def _make_key(
    query: str,
    topic: str | list[str] | None,
    uin: str | None,
    extra: dict | None = None,
) -> str:
    """Hash every input that changes retrieval results."""
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
    """Return cached hits, or None when absent or expired."""
    key = _make_key(query, topic, uin, extra)
    entry = _entries.get(key)
    if entry is None:
        return None

    expires_at, hits = entry
    if time.time() >= expires_at:
        _entries.pop(key, None)
        return None

    logger.info("rag_cache_hit query=%r", query[:50])
    return hits


def set(
    query: str,
    hits: list[dict],
    topic: str | list[str] | None = None,
    uin: str | None = None,
    extra: dict | None = None,
) -> None:
    """Store hits for the configured TTL."""
    key = _make_key(query, topic, uin, extra)
    _entries[key] = (time.time() + settings.rag_cache_ttl_seconds, hits)
    logger.info("rag_cache_store query=%r entries=%s", query[:60], len(_entries))
