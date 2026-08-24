"""Wipe the Pinecone indexes and rebuild them with the new pipeline.

Run it with:  uv run python scripts/reindex.py
Skip the confirmation prompt by adding --yes

The project's packages currently live in venv/ rather than uv's default
.venv/, so point uv at it first:
    set UV_PROJECT_ENVIRONMENT=venv        (Windows cmd)
    $env:UV_PROJECT_ENVIRONMENT="venv"     (PowerShell)

WHY a wipe: the old index was built by the sliding-window chunker. Leaving
those vectors in place would mean every search competes against chunks that
cut mid-clause, so the whole namespace has to go.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly, not just as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.ingestion.pipeline import run_pipeline  # noqa: E402
from src.utils import configure_logging  # noqa: E402


def _namespace_counts(index) -> dict:
    """Return {namespace: vector_count} for one index."""
    try:
        stats = index.describe_index_stats()
    except Exception as error:
        print(f"    could not read stats: {error}")
        return {}

    namespaces = stats.get("namespaces") or {}
    counts = {}
    for name, info in namespaces.items():
        # The SDK returns either a dict or an object depending on version.
        count = info.get("vector_count") if isinstance(info, dict) else getattr(
            info, "vector_count", 0
        )
        counts[name] = int(count or 0)
    return counts


def _clear_index(index, label: str) -> None:
    """Delete every vector in EVERY namespace of one index.

    WHY all namespaces and not just ours: an earlier run wrote 6,446 vectors
    into the default namespace instead of the configured one. Clearing only
    the configured namespace left those behind, so the index kept answering
    from chunks built by the old chunker with the old topic tags.
    """
    before = _namespace_counts(index)
    total = sum(before.values())
    print(f"  {label}: {total} vectors across {len(before)} namespace(s) -> {before}")

    for namespace in before:
        try:
            index.delete(delete_all=True, namespace=namespace)
            print(f"    cleared namespace '{namespace}'")
        except Exception as error:
            # A namespace that no longer exists raises 404 -- nothing to do.
            print(f"    namespace '{namespace}': nothing to delete ({error})")

    after = _namespace_counts(index)
    print(f"  {label}: {sum(after.values())} vectors after")


def clear_pinecone() -> None:
    """Empty both the dense and the hosted sparse index completely."""
    from pinecone import Pinecone

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required to re-index")

    client = Pinecone(api_key=settings.pinecone_api_key)
    existing = client.list_indexes().names()

    print("Clearing every namespace in both indexes...")
    for name, label in (
        (settings.dense_index_name, "dense"),
        (settings.sparse_index_name, "sparse"),
    ):
        if name not in existing:
            print(f"  {label}: index '{name}' does not exist yet, skipping")
            continue
        _clear_index(client.Index(name), label)


def confirm(skip_prompt: bool) -> bool:
    """Ask before destroying the index, unless --yes was passed."""
    if skip_prompt:
        return True

    print("")
    print("This DELETES every vector in Pinecone namespace "
          f"'{settings.namespace}' and rebuilds it from scratch.")
    answer = input("Type 'yes' to continue: ").strip().lower()
    return answer == "yes"


def main() -> None:
    """Clear Pinecone, then run the ingestion pipeline over all PDFs."""
    configure_logging()

    if not confirm("--yes" in sys.argv):
        print("Aborted -- nothing was deleted.")
        return

    clear_pinecone()
    print("")
    run_pipeline(push_to_pinecone=True)
    print("")
    print("Re-index finished.")


if __name__ == "__main__":
    main()
