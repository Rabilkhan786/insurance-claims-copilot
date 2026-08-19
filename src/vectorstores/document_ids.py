"""Deterministic identifiers shared by dense and sparse Pinecone records."""
from hashlib import sha256

FIELD_SEPARATOR = "\x1f"


def document_id(document) -> str:
    """Return a stable ID based on a document's content and source metadata."""
    metadata = document.metadata
    identity_fields = (
        str(metadata.get("source", "")),
        str(metadata.get("page_number", "")),
        str(metadata.get("category", "")),
        document.page_content,
    )
    identity = FIELD_SEPARATOR.join(identity_fields)
    return f"doc-{sha256(identity.encode('utf-8')).hexdigest()}"
