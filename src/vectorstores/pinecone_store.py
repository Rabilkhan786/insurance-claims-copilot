"""Adapters for the existing dense and hosted sparse Pinecone indexes.

WHY this is not langchain-pinecone's PineconeVectorStore: that class covers
one dense index, and this project runs two indexes of different kinds. The
sparse half is a Pinecone *hosted* sparse index -- created with
create_index_for_model() and queried with search(inputs={"text": ...}), so
Pinecone does the sparse encoding server-side. The official integration has
no route to that API, so adopting it would mean dropping the lexical half of
hybrid retrieval, or rebuilding a BM25 index in the API process on every
boot and letting it drift from what was actually indexed.

Everything above this layer *is* framework code: the two thin retrievers in
src/retrieval/retrievers.py wrap these methods as LangChain BaseRetrievers,
and EnsembleRetriever and CrossEncoderReranker take it from there. This
module is only the part the framework does not reach.
"""
import logging
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from config import settings

from .document_ids import document_id

logger = logging.getLogger(__name__)

# Pinecone rejects an upsert with a 5xx or a timeout often enough over a
# multi-thousand-vector run that one failure should not lose the batch.
# tenacity owns the backoff -- the hand-written sleep loop this replaced was
# deleted in a refactor while its call sites stayed, which left the whole
# indexing path raising NameError on the first upsert.
_retry_upsert = retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)


class PineconeHybridStore:
    """Read and write the dense and hosted sparse Pinecone indexes."""

    def __init__(self, embedder) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is required for retrieval")

        from pinecone import Pinecone

        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.dense_index = self.client.Index(settings.dense_index_name)
        self.sparse_index = self.client.Index(settings.sparse_index_name)
        self.embedder = embedder

    def ensure_indexes(self) -> None:
        """Create the dense and hosted sparse indexes if they don't exist."""
        from pinecone import ServerlessSpec

        existing = self.client.list_indexes().names()

        if settings.dense_index_name not in existing:
            specification = ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            )
            self.client.create_index(
                name=settings.dense_index_name,
                dimension=settings.dense_dimension,
                metric=settings.dense_metric,
                spec=specification,
            )

        if settings.sparse_index_name not in existing:
            embedding_config = {
                "model": settings.sparse_model,
                "field_map": {"text": "chunk_text"},
            }
            self.client.create_index_for_model(
                name=settings.sparse_index_name,
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
                embed=embedding_config,
            )

        self.dense_index = self.client.Index(settings.dense_index_name)
        self.sparse_index = self.client.Index(settings.sparse_index_name)

    def index_documents(self, documents: list) -> None:
        """Upsert matching document IDs into the two independent indexes.

        embed_documents returns plain lists of floats (the LangChain
        Embeddings contract), which is already what Pinecone's upsert wants.
        """
        texts = [document.page_content for document in documents]
        embeddings = self.embedder.embed_documents(texts)

        for start in range(0, len(documents), settings.batch_size):
            batch_documents = documents[start:start + settings.batch_size]
            batch_vectors = embeddings[start:start + settings.batch_size]

            dense_records = []
            sparse_records = []

            for document, embedding in zip(batch_documents, batch_vectors):
                identifier = document_id(document)
                metadata = {
                    "text": document.page_content,
                    **document.metadata,
                }

                dense_records.append(
                    {
                        "id": identifier,
                        "values": list(embedding),
                        "metadata": metadata,
                    }
                )
                sparse_records.append(
                    {
                        "id": identifier,
                        "chunk_text": document.page_content,
                        **document.metadata,
                    }
                )

            _retry_upsert(self.dense_index.upsert)(
                vectors=dense_records,
                namespace=settings.namespace,
            )
            _retry_upsert(self.sparse_index.upsert_records)(
                namespace=settings.namespace,
                records=sparse_records,
            )
            print(f"    upserted {start + len(batch_documents)}/{len(documents)}")

    def dense_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        embedding = self.embedder.embed_query(query)
        response = self.dense_index.query(
            vector=embedding,
            top_k=settings.dense_top_k,
            include_metadata=True,
            namespace=settings.namespace,
            filter=metadata_filter,
        )
        return [
            {
                "id": item.id,
                "score": item.score,
                "metadata": dict(item.metadata or {}),
            }
            for item in response.matches
        ]

    def sparse_search(
        self,
        query: str,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        response = self.sparse_index.search(
            namespace=settings.namespace,
            top_k=settings.sparse_top_k,
            inputs={"text": query},
            filter=metadata_filter,
        )
        return [
            {
                "id": item.id,
                "score": item.score,
                "metadata": dict(item.fields or {}),
            }
            for item in response.result.hits
        ]
