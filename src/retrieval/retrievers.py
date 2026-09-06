"""Hybrid dense + sparse retrieval with cross-encoder reranking."""

from functools import lru_cache
from typing import Any

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from config import settings
from src.embeddings import get_embedder
from src.vectorstores import PineconeHybridStore

ID_KEY = "doc_id"


def _to_documents(hits: list[dict]) -> list[Document]:
    documents = []
    for hit in hits:
        metadata = dict(hit.get("metadata") or {})
        text = metadata.get("text") or metadata.get("chunk_text") or ""
        metadata[ID_KEY] = hit.get("id")
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


class _PineconeRetriever(BaseRetriever):
    """Base retriever used by the dense and sparse searches."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    store: Any
    metadata_filter: dict[str, Any] | None = None
    search_method: str = "dense_search"

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        search = getattr(self.store, self.search_method)
        return _to_documents(search(query, self.metadata_filter))


class PineconeDenseRetriever(_PineconeRetriever):
    search_method: str = "dense_search"


class PineconeSparseRetriever(_PineconeRetriever):
    search_method: str = "sparse_search"


@lru_cache(maxsize=1)
def get_store() -> PineconeHybridStore:
    return PineconeHybridStore(get_embedder())


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model=HuggingFaceCrossEncoder(model_name=settings.cross_encoder_model),
        top_n=settings.final_top_k,
    )


def build_retriever(metadata_filter: dict[str, Any] | None = None):
    fused = EnsembleRetriever(
        retrievers=[
            PineconeDenseRetriever(store=get_store(), metadata_filter=metadata_filter),
            PineconeSparseRetriever(store=get_store(), metadata_filter=metadata_filter),
        ],
        weights=[0.5, 0.5],
        c=settings.rrf_k,
        id_key=ID_KEY,
    )
    return ContextualCompressionRetriever(
        base_compressor=get_reranker(),
        base_retriever=fused,
    )


def retrieve(query: str, metadata_filter: dict[str, Any] | None = None) -> list[Document]:
    documents = build_retriever(metadata_filter).invoke(query)
    return documents


def warmup() -> None:
    get_store()
    get_reranker().model.score([("warmup", "warmup")])
