"""Root entry point for the PDF-to-Pinecone indexing workflow."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from config import settings
from src.embeddings import BGEEmbedder
from src.ingestion import load_pdf_documents
from src.utils import configure_logging
from src.vectorstores import PineconeHybridStore


logger = logging.getLogger(__name__)


def save_indexing_manifest(document_count: int) -> None:
    """Save the latest indexing run's result in the artifacts folder."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "indexed_at": datetime.now(UTC).isoformat(),
        "document_count": document_count,
        "embedding_model": settings.embedding_model,
        "dense_index": settings.dense_index_name,
        "sparse_index": settings.sparse_index_name,
        "namespace": settings.namespace,
    }

    manifest_path = settings.artifacts_dir / "indexing_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """Load PDFs, embed them, and upload dense and sparse index records."""
    configure_logging()

    documents = load_pdf_documents()
    if not documents:
        raise RuntimeError(
            f"No eligible PDF chunks found in {settings.data_dir}"
        )

    embedder = BGEEmbedder()
    vector_store = PineconeHybridStore(embedder)

    vector_store.ensure_indexes()
    vector_store.index_documents(documents)
    save_indexing_manifest(len(documents))

    manifest_path = settings.artifacts_dir / "indexing_manifest.json"
    logger.info("indexing_complete document_count=%s", len(documents))
    logger.info("dense_index=%s", settings.dense_index_name)
    logger.info("sparse_index=%s", settings.sparse_index_name)
    logger.info("artifact=%s", manifest_path)


if __name__ == "__main__":
    main()
