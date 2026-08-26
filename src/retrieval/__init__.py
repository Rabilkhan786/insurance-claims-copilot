"""Public exports for the hybrid retrieval subpackage.

Fusion and reranking are composed from LangChain retrievers in retrievers.py
rather than implemented here -- see that module for why. Only the two
functions callers actually use are exported; the retriever classes are an
implementation detail of build_retriever().
"""
from .retrievers import retrieve, warmup

__all__ = ["retrieve", "warmup"]
