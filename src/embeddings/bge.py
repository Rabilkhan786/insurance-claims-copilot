"""BGE embedding model used by the dense Pinecone index."""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """Return the shared normalized BGE embedder."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.batch_size,
        },
    )
