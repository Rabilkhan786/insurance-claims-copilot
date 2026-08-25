"""Root entry point for the PDF-to-Pinecone indexing workflow.

Runs the section-aware pipeline (src/ingestion/pipeline.py) over every PDF
in Data/insurance_documents/ and upserts the result into Pinecone.
Upserts are idempotent -- document IDs hash the chunk's own content -- so
re-running overwrites the same vectors rather than duplicating them.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from config import settings
from src.ingestion import run_pipeline
from src.utils import configure_logging


logger = logging.getLogger(__name__)


def save_indexing_manifest(counters: dict) -> None:
    """Save the latest indexing run's result in the artifacts folder."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "indexed_at": datetime.now(UTC).isoformat(),
        "pdf_count": counters["pdfs"],
        "document_count": counters["documents"],
        "text_chunks": counters["chunks"],
        "table_sentences": counters["table_sentences"],
        "sql_rows_staged": counters["sql_rows"],
        "tables_skipped": counters["tables_skipped"],
        "tables_by_type": counters["tables_by_type"],
        "chunk_strategy": settings.chunk_strategy,
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


def clear_index() -> None:
    """Delete every vector in the namespace, in both indexes.

    WHY this is needed: a document ID is a hash of the chunk's own content, so
    re-running the pipeline overwrites the chunks it produces this time. It has
    no way to know about chunks it produced last time from a PDF that has since
    been removed -- those keep answering questions from a document that is no
    longer part of the corpus. Clearing first is the only way to drop them.
    """
    from pinecone import Pinecone

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to clear the index")

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing = client.list_indexes().names()

    for name in (settings.dense_index_name, settings.sparse_index_name):
        if name not in existing:
            continue
        try:
            client.Index(name).delete(delete_all=True, namespace=settings.namespace)
            print(f"  cleared '{settings.namespace}' in {name}")
        except Exception as error:
            # A namespace that was already empty raises 404 -- nothing to drop.
            print(f"  {name}: nothing to clear ({error})")


def main() -> None:
    """Parse every policy PDF, embed the chunks, and upload them."""
    configure_logging()

    if "--reset" in sys.argv:
        print("Clearing the index before rebuilding...")
        clear_index()

    counters = run_pipeline(push_to_pinecone=True)
    if not counters["documents"]:
        raise RuntimeError(
            f"No eligible PDF chunks found in {settings.data_dir}"
        )

    save_indexing_manifest(counters)

    manifest_path = settings.artifacts_dir / "indexing_manifest.json"
    logger.info("indexing_complete document_count=%s", counters["documents"])
    logger.info("dense_index=%s", settings.dense_index_name)
    logger.info("sparse_index=%s", settings.sparse_index_name)
    logger.info("artifact=%s", manifest_path)


if __name__ == "__main__":
    main()
