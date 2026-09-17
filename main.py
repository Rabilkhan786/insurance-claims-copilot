"""Index policy PDFs into Pinecone.

Run:
    uv run python main.py
    uv run python main.py --reset
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime

from config import settings
from src.ingestion import run_pipeline
from src.utils import configure_logging

logger = logging.getLogger(__name__)


def save_indexing_manifest(counters: dict) -> None:
    """Save a summary of the latest indexing run."""
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
    """Remove all vectors from the configured Pinecone namespace."""
    from pinecone import Pinecone

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to clear the index")

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing = client.list_indexes().names()

    for name in (settings.dense_index_name, settings.sparse_index_name):
        if name not in existing:
            continue

        client.Index(name).delete(
            delete_all=True,
            namespace=settings.namespace,
        )
        print(f"  cleared '{settings.namespace}' in {name}")


def main() -> None:
    """Parse policy PDFs and upload their chunks to Pinecone."""
    configure_logging()

    parser = argparse.ArgumentParser(description="Index policy PDFs into Pinecone")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear the namespace before indexing",
    )
    args = parser.parse_args()

    if args.reset:
        print("Clearing the index before rebuilding...")
        clear_index()

    counters = run_pipeline(push_to_pinecone=True)
    if not counters["documents"]:
        raise RuntimeError(f"No eligible PDF chunks found in {settings.data_dir}")

    save_indexing_manifest(counters)

    manifest_path = settings.artifacts_dir / "indexing_manifest.json"
    logger.info("indexing_complete document_count=%s", counters["documents"])
    logger.info("dense_index=%s", settings.dense_index_name)
    logger.info("sparse_index=%s", settings.sparse_index_name)
    logger.info("artifact=%s", manifest_path)


if __name__ == "__main__":
    main()
