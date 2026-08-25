"""Tests for deterministic document ID generation."""
from types import SimpleNamespace

from src.vectorstores.document_ids import document_id


def _doc(text="some chunk text", source="policy.pdf", page_number=1, category="NarrativeText"):
    return SimpleNamespace(
        page_content=text,
        metadata={"source": source, "page_number": page_number, "category": category},
    )


def test_same_content_produces_same_id():
    doc_a = _doc()
    doc_b = _doc()
    assert document_id(doc_a) == document_id(doc_b)


def test_different_text_produces_different_id():
    doc_a = _doc(text="text one")
    doc_b = _doc(text="text two")
    assert document_id(doc_a) != document_id(doc_b)


def test_different_source_produces_different_id():
    doc_a = _doc(source="a.pdf")
    doc_b = _doc(source="b.pdf")
    assert document_id(doc_a) != document_id(doc_b)


def test_id_has_expected_prefix():
    assert document_id(_doc()).startswith("doc-")


def test_missing_metadata_does_not_raise():
    doc = SimpleNamespace(page_content="text", metadata={})
    assert document_id(doc).startswith("doc-")
