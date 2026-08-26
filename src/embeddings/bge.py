"""The embedding model behind the dense Pinecone index.

WHY there is no wrapper class here any more: this module used to define a
BGEEmbedder that held a SentenceTransformer and exposed embed_documents /
embed_query -- which is precisely the LangChain Embeddings interface, and
precisely what langchain-huggingface's HuggingFaceEmbeddings already
implements. Keeping our own copy cost a class here plus a second adapter
class in the evaluation runner, whose only job was to convert this one back
into the interface it had reimplemented.

Normalised embeddings matter: the dense index uses cosine similarity, and
normalising at encode time is what makes the scores comparable.
"""
from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """Return the shared BGE embedder, loading its weights once per process."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": settings.batch_size,
        },
    )
