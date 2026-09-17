"""Run PDF ingestion for policy documents."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from config import settings

from .chunker import chunk_page, last_topics
from .page_parser import parse_page
from .table_classifier import (
    PINECONE_ONLY,
    SKIP,
    SQL_AND_PINECONE,
    SQL_ONLY,
    classify_table,
    retrieval_topics,
)
from .table_converter import table_to_rows, table_to_sentences

logger = logging.getLogger(__name__)

UIN_PATTERN = re.compile(r"\b[A-Z]{5,8}\d{4,5}V\d{6}\b")
LEGACY_UIN_PATTERN = re.compile(r"\bIRDAI{0,2}/[A-Z0-9()./\-]{5,50}")


def extract_uin(pdf_name: str, sample_text: str) -> str:
    """Return a policy UIN when present; otherwise return an empty string."""
    for text in (pdf_name, sample_text):
        match = UIN_PATTERN.search(text.upper())
        if match:
            return match.group(0)

    legacy_match = LEGACY_UIN_PATTERN.search(sample_text.upper())
    if legacy_match:
        return legacy_match.group(0).rstrip("./-)")

    return ""


def sample_pages_for_uin(pdf, page_limit: int = 4) -> str:
    """Read a few pages for optional UIN extraction."""
    indexes = list(range(min(page_limit, pdf.page_count)))
    last_page = pdf.page_count - 1

    if pdf.page_count and last_page not in indexes:
        indexes.append(last_page)

    return "\n".join(pdf[index].get_text() for index in indexes)


def extract_product(pdf_name: str, pdf_metadata: dict | None = None) -> str:
    """Return the PDF title or a readable version of its filename."""
    metadata = pdf_metadata or {}
    title = " ".join(str(metadata.get("title") or "").split())
    if title:
        return title[:160]

    stem = Path(pdf_name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()


def extract_insurer(pdf_metadata: dict | None = None) -> str:
    """Return insurer metadata only when the PDF explicitly identifies it."""
    metadata = pdf_metadata or {}
    author = " ".join(str(metadata.get("author") or "").split()).strip()
    if not author:
        return ""

    lowered = author.lower()
    software_authors = (
        "microsoft word",
        "adobe acrobat",
        "acrobat pdfmaker",
    )
    if any(name in lowered for name in software_authors):
        return ""

    # PDF author metadata is not guaranteed to be the insurer. Keep it only
    # when it explicitly looks like an insurance organization.
    if "insurance" not in lowered and "assurance" not in lowered:
        return ""

    return author[:120]


def _build_document(text: str, metadata: dict, pdf_name: str):
    """Create one LangChain document with stable retrieval metadata."""
    from langchain_core.documents import Document

    document_metadata = {
        **metadata,
        "source": pdf_name,
        "page_number": metadata["page"],
        "category": metadata["chunk_type"],
    }
    return Document(page_content=text, metadata=document_metadata)


def _table_documents(
    table: dict,
    table_type: str,
    context: dict,
) -> list:
    """Convert one extracted table into retrieval documents."""
    topics = retrieval_topics(table_type)
    sentences = table_to_sentences(
        table,
        table_type,
        context["insurer"],
        context["uin"],
        context["page"],
    )

    metadata = {
        "uin": context["uin"],
        "insurer": context["insurer"],
        "product": context["product"],
        "page": context["page"],
        "section": table_type,
        "topic": topics[0],
        "topics": topics,
        "chunk_type": "table_sentence",
    }

    return [
        _build_document(sentence, metadata, context["pdf_name"])
        for sentence in sentences
    ]


def _send_to_pinecone(
    table: dict,
    table_type: str,
    context: dict,
    counters: dict,
    documents: list,
) -> None:
    """Add table retrieval documents to the current batch."""
    table_documents = _table_documents(table, table_type, context)
    documents.extend(table_documents)
    counters["table_sentences"] += len(table_documents)


def _stage_for_sql(
    table: dict,
    table_type: str,
    context: dict,
    counters: dict,
    staged_rows: list[dict],
) -> None:
    """Stage recognized table rows for optional structured storage."""
    rows = table_to_rows(table, table_type)

    for row in rows:
        staged_rows.append(
            {
                "uin": context["uin"],
                "insurer": context["insurer"],
                "product": context["product"],
                "source": context["pdf_name"],
                "page": context["page"],
                "table_type": table_type,
                "row": row,
            }
        )

    counters["sql_rows"] += len(rows)


def _route_table(
    table: dict,
    context: dict,
    counters: dict,
    documents: list,
    staged_rows: list[dict],
) -> str | None:
    """Classify one table and route it to RAG, SQL staging, or both."""
    verdict = classify_table(table)
    table_type = verdict["table_type"]
    destination = verdict["destination"]

    counters["tables_by_type"][table_type] = (
        counters["tables_by_type"].get(table_type, 0) + 1
    )

    if destination == SKIP:
        counters["tables_skipped"] += 1
        return None

    if destination in (SQL_AND_PINECONE, PINECONE_ONLY):
        _send_to_pinecone(
            table,
            table_type,
            context,
            counters,
            documents,
        )

    if destination in (SQL_AND_PINECONE, SQL_ONLY):
        _stage_for_sql(
            table,
            table_type,
            context,
            counters,
            staged_rows,
        )

    return f"{table_type}->{destination}"


def _process_one_page(
    page,
    context: dict,
    counters: dict,
    documents: list,
    staged_rows: list[dict],
) -> list[str] | None:
    """Process the prose and tables from one PDF page."""
    prose, tables = parse_page(page)

    chunks = chunk_page(
        prose,
        context["uin"],
        context["insurer"],
        context["product"],
        context["page"],
        context.get("carry_topics"),
    )

    for chunk in chunks:
        documents.append(
            _build_document(
                chunk["text"],
                chunk["metadata"],
                context["pdf_name"],
            )
        )

    counters["chunks"] += len(chunks)

    routes = []
    for table in tables:
        route = _route_table(
            table,
            context,
            counters,
            documents,
            staged_rows,
        )
        if route:
            routes.append(route)

    if chunks or routes:
        table_note = f" | tables: {', '.join(routes)}" if routes else ""
        print(
            f"    page {context['page']:>3}: "
            f"{len(chunks):>2} chunks{table_note}"
        )

    return last_topics(chunks) or context.get("carry_topics")


def process_pdf(pdf_path: Path, counters: dict) -> tuple[list, list[dict]]:
    """Convert one readable PDF into RAG documents and structured rows."""
    import pymupdf

    documents = []
    staged_rows = []

    with pymupdf.open(pdf_path) as pdf:
        metadata = pdf.metadata or {}
        uin = extract_uin(pdf_path.name, sample_pages_for_uin(pdf))
        insurer = extract_insurer(metadata)
        product = extract_product(pdf_path.name, metadata)

        print(f"\n  {pdf_path.name}")
        print(
            f"    product={product} | uin={uin or 'not found'} "
            f"| pages={pdf.page_count}"
        )

        if not uin:
            counters["pdfs_without_uin"] += 1
            logger.info("pdf_without_uin name=%s", pdf_path.name)

        carry_topics = None
        for page_index in range(pdf.page_count):
            context = {
                "uin": uin,
                "insurer": insurer,
                "product": product,
                "page": page_index + 1,
                "pdf_name": pdf_path.name,
                "carry_topics": carry_topics,
            }
            carry_topics = _process_one_page(
                pdf[page_index],
                context,
                counters,
                documents,
                staged_rows,
            )

    counters["pdfs"] += 1
    return documents, staged_rows


def new_counters() -> dict:
    """Create counters for one ingestion run."""
    return {
        "pdfs": 0,
        "pdfs_without_uin": 0,
        "chunks": 0,
        "table_sentences": 0,
        "sql_rows": 0,
        "tables_skipped": 0,
        "tables_by_type": {},
    }


def save_staged_tables(staged_rows: list[dict]) -> Path:
    """Write staged structured rows to the artifacts directory."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifacts_dir / "staged_tables.json"
    path.write_text(json.dumps(staged_rows, indent=2), encoding="utf-8")
    return path


def print_summary(counters: dict, staged_path: Path) -> None:
    """Print a compact summary of one ingestion run."""
    print("")
    print("=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    print(f"  PDFs processed      : {counters['pdfs']}")
    print(f"  PDFs without UIN    : {counters['pdfs_without_uin']}")
    print(f"  Text chunks         : {counters['chunks']}")
    print(f"  Table sentences     : {counters['table_sentences']}")
    print(f"  Rows staged for SQL : {counters['sql_rows']}")
    print(f"  Staged rows file    : {staged_path}")
    print(f"  Tables skipped      : {counters['tables_skipped']}")
    print("  Tables by type:")

    for table_type, count in sorted(
        counters["tables_by_type"].items(),
        key=lambda item: -item[1],
    ):
        print(f"    {table_type:<22} {count}")

    print("=" * 60)


def _collect_documents_from_pdfs(
    pdf_paths: list[Path],
    counters: dict,
) -> tuple[list, list[dict]]:
    """Process all PDFs while allowing one bad file to fail independently."""
    documents = []
    staged_rows = []

    for pdf_path in pdf_paths:
        try:
            pdf_documents, pdf_rows = process_pdf(pdf_path, counters)
        except Exception:
            logger.exception("pdf_failed path=%s", pdf_path)
            print(f"    -- failed, skipping {pdf_path.name}")
            continue

        documents.extend(pdf_documents)
        staged_rows.extend(pdf_rows)

    return documents, staged_rows


def run_pipeline(push_to_pinecone: bool = True) -> dict:
    """Process every PDF under the configured policy-document directory."""
    pdf_paths = sorted(settings.data_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {settings.data_dir}")

    print(f"Found {len(pdf_paths)} PDFs in {settings.data_dir}")
    counters = new_counters()

    documents, staged_rows = _collect_documents_from_pdfs(
        pdf_paths,
        counters,
    )
    staged_path = save_staged_tables(staged_rows)

    if push_to_pinecone and documents:
        print(f"\nEmbedding and upserting {len(documents)} documents...")

        from src.embeddings import get_embedder
        from src.vectorstores import PineconeHybridStore

        store = PineconeHybridStore(get_embedder())
        store.ensure_indexes()
        store.index_documents(documents)
        print("Upsert complete.")

    print_summary(counters, staged_path)
    counters["documents"] = len(documents)
    return counters
