"""Dependency-light evaluation metrics kept outside the runtime application.

These are simplified keyword/token-overlap approximations of retrieval and
answer quality, not the LLM-judged RAGAS metrics of the same name. They are
useful as a fast, offline sanity check, not as a substitute for RAGAS-style
evaluation.
"""
from collections.abc import Iterable

MIN_SIGNIFICANT_WORD_LENGTH = 3
PUNCTUATION_TO_STRIP = ".,!?;:\"'()[]{}"


def keyword_context_precision(
    retrieved: Iterable[str],
    expected_keywords: Iterable[str],
) -> float:
    """Fraction of retrieved chunks containing at least one keyword."""
    documents = list(retrieved)
    keywords = [keyword.lower() for keyword in expected_keywords]

    if not documents:
        return 0.0

    matches = sum(
        any(keyword in document.lower() for keyword in keywords)
        for document in documents
    )
    return matches / len(documents)


def keyword_context_recall(
    retrieved: Iterable[str],
    expected_keywords: Iterable[str],
) -> float:
    """Fraction of expected keywords found anywhere in retrieved chunks."""
    documents = list(retrieved)
    keywords = [keyword.lower() for keyword in expected_keywords]

    if not keywords:
        return 0.0

    corpus = " ".join(documents).lower()
    matches = sum(keyword in corpus for keyword in keywords)
    return matches / len(keywords)


def token_overlap_faithfulness(
    answer: str,
    contexts: Iterable[str],
) -> float:
    """Fraction of the answer's significant words present in the context."""
    answer_tokens = _significant_words(answer)
    context_tokens = _significant_words(" ".join(contexts))
    overlap = answer_tokens & context_tokens
    return len(overlap) / max(len(answer_tokens), 1)


def token_overlap_relevancy(answer: str, question: str) -> float:
    """Fraction of the question's significant words present in the answer."""
    question_tokens = _significant_words(question)
    answer_tokens = _significant_words(answer)
    overlap = question_tokens & answer_tokens
    return len(overlap) / max(len(question_tokens), 1)


def _significant_words(text: str) -> set[str]:
    """Lowercase, punctuation-stripped words longer than the minimum length."""
    words = (word.strip(PUNCTUATION_TO_STRIP) for word in text.split())
    return {
        word.lower()
        for word in words
        if len(word) > MIN_SIGNIFICANT_WORD_LENGTH
    }
