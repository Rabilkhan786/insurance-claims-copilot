"""The notebook's deterministic reciprocal-rank fusion implementation."""
from collections import defaultdict
from typing import Any


def reciprocal_rank_fusion(
    dense_hits: list[dict[str, Any]],
    sparse_hits: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """Merge dense and sparse hit lists into one ranked list using RRF."""
    scores: defaultdict[str, float] = defaultdict(float)
    documents: dict[str, dict[str, Any]] = {}

    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            identifier = hit["id"]
            scores[identifier] += 1 / (k + rank)
            documents.setdefault(identifier, hit)

    ranked_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        {**documents[identifier], "rrf_score": score}
        for identifier, score in ranked_ids
    ]
