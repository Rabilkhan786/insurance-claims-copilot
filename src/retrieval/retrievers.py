"""Hybrid retrieval, composed from LangChain retrievers instead of by hand.

The shape is the one the reference book describes -- run a dense and a sparse
retriever in parallel, fuse their rankings, then rerank the survivors with a
cross-encoder -- but built with the current LangChain 1.x classes:

    EnsembleRetriever              fuses the two rankings (weighted RRF)
      wrapped in
    ContextualCompressionRetriever reranks and trims to final_top_k

WHY this replaced the hand-written version: src/retrieval/rrf.py implemented
reciprocal rank fusion by hand with a k constant of 60, which is exactly what
EnsembleRetriever already does (its `c` field defaults to 60). Keeping our own
copy meant maintaining code the framework ships and tests for us.

Pinecone is still queried through the project's own PineconeHybridStore -- the
thin retrievers below only adapt its dicts into LangChain Documents, so the
metadata filter that narrows the search still happens Pinecone-side.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config import settings
from src.embeddings import BGEEmbedder
from src.vectorstores import PineconeHybridStore

logger = logging.getLogger(__name__)

# EnsembleRetriever needs a stable key to tell "the same chunk from both
# retrievers" apart from two different chunks, otherwise it dedupes on the
# full page_content string.
ID_KEY = "doc_id"


def _to_documents(hits: list[dict]) -> list[Document]:
    """Adapt Pinecone hit dicts into LangChain Documents.

    The dense index stores the chunk under "text" and the hosted sparse index
    under "chunk_text", so both spellings are checked here rather than in
    every caller.
    """
    documents = []
    for hit in hits:
        metadata = dict(hit.get("metadata") or {})
        text = metadata.get("text") or metadata.get("chunk_text") or ""
        metadata[ID_KEY] = hit.get("id")
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


class PineconeDenseRetriever(BaseRetriever):
    """Semantic half of the hybrid: BGE embeddings against the dense index."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any
    metadata_filter: dict[str, Any] | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return _to_documents(self.store.dense_search(query, self.metadata_filter))


class PineconeSparseRetriever(BaseRetriever):
    """Lexical half of the hybrid: Pinecone's hosted sparse index.

    This is used instead of the book's in-memory BM25Retriever because the
    corpus already lives in Pinecone -- rebuilding a BM25 index in the API
    process on every boot would be slower and would drift from what was
    actually indexed.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Any
    metadata_filter: dict[str, Any] | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return _to_documents(self.store.sparse_search(query, self.metadata_filter))


_store: PineconeHybridStore | None = None
_reranker: CrossEncoderReranker | None = None


def get_store() -> PineconeHybridStore:
    """Return the shared Pinecone store, building it once per process."""
    global _store
    if _store is None:
        _store = PineconeHybridStore(BGEEmbedder())
    return _store


def get_reranker() -> CrossEncoderReranker:
    """Return the shared cross-encoder, loading its weights once per process."""
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoderReranker(
            model=HuggingFaceCrossEncoder(model_name=settings.cross_encoder_model),
            top_n=settings.final_top_k,
        )
    return _reranker


def build_retriever(metadata_filter: dict[str, Any] | None = None):
    """Compose the full retrieval chain for one metadata filter.

    The filter changes per question, and a LangChain retriever takes its
    configuration at construction time, so the small wrapper objects are
    rebuilt per call. That is cheap: the expensive parts -- the embedding
    model and the cross-encoder -- are the cached singletons above.
    """
    store = get_store()
    fused = EnsembleRetriever(
        retrievers=[
            PineconeDenseRetriever(store=store, metadata_filter=metadata_filter),
            PineconeSparseRetriever(store=store, metadata_filter=metadata_filter),
        ],
        # Equal weights: neither half is trusted more than the other, which
        # matches the plain RRF the hand-written version did.
        weights=[0.5, 0.5],
        c=settings.rrf_k,
        id_key=ID_KEY,
    )
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(),
        base_retriever=fused,
    )


def retrieve(query: str, metadata_filter: dict[str, Any] | None = None) -> list[Document]:
    """Run one filtered hybrid search and return the reranked Documents."""
    documents = build_retriever(metadata_filter).invoke(query)
    logger.info(
        "retrieved query=%r filter=%s results=%s",
        query, metadata_filter, len(documents),
    )
    return documents


def warmup() -> None:
    """Load the embedder and cross-encoder before the first real request.

    Constructing the models loads their weights, but the first predict() still
    pays a one-off graph/kernel cost. Paying it here keeps it off the first
    claim an employee submits.
    """
    get_store()
    get_reranker().model.score([("warmup", "warmup")])
