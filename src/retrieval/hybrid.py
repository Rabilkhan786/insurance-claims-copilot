"""Orchestrates dense search, sparse search, RRF, and reranking."""
import logging
import time

from config import settings

from .rrf import reciprocal_rank_fusion

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combine dense and sparse retrieval before cross-encoder reranking."""

    def __init__(self, store, reranker) -> None:
        self.store = store
        self.reranker = reranker

    def retrieve(self, query: str) -> list[dict]:
        started_at = time.perf_counter()
        dense_hits = self.store.dense_search(query)
        logger.info(
            "dense_retrieval_seconds=%.3f",
            time.perf_counter() - started_at,
        )

        started_at = time.perf_counter()
        sparse_hits = self.store.sparse_search(query)
        logger.info(
            "sparse_retrieval_seconds=%.3f",
            time.perf_counter() - started_at,
        )

        started_at = time.perf_counter()
        fused_hits = reciprocal_rank_fusion(
            dense_hits,
            sparse_hits,
            settings.rrf_k,
        )
        logger.info("rrf_seconds=%.3f", time.perf_counter() - started_at)

        started_at = time.perf_counter()
        reranked_hits = self.reranker.rerank(
            query,
            fused_hits[:settings.rrf_top_k],
        )
        logger.info(
            "cross_encoder_seconds=%.3f",
            time.perf_counter() - started_at,
        )
        return reranked_hits
