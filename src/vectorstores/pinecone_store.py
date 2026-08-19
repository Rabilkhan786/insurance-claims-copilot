"""Adapters for the existing dense and hosted sparse Pinecone indexes."""
from typing import Any

from config import settings

from .document_ids import document_id


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
        """Upsert matching document IDs into the two independent indexes."""
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
                        "values": embedding.tolist(),
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

            self.dense_index.upsert(
                vectors=dense_records,
                namespace=settings.namespace,
            )
            self.sparse_index.upsert_records(
                namespace=settings.namespace,
                records=sparse_records,
            )

    def dense_search(self, query: str) -> list[dict[str, Any]]:
        embedding = self.embedder.embed_query(query)
        response = self.dense_index.query(
            vector=embedding,
            top_k=settings.dense_top_k,
            include_metadata=True,
            namespace=settings.namespace,
        )
        return [
            {
                "id": item.id,
                "score": item.score,
                "metadata": dict(item.metadata or {}),
            }
            for item in response.matches
        ]

    def sparse_search(self, query: str) -> list[dict[str, Any]]:
        search_query = {
            "top_k": settings.sparse_top_k,
            "inputs": {"text": query},
        }
        response = self.sparse_index.search(
            namespace=settings.namespace,
            query=search_query,
        )
        return [
            {
                "id": item.id,
                "score": item.score,
                "metadata": dict(item.fields or {}),
            }
            for item in response.result.hits
        ]
