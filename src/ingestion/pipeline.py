"""Parse policy PDFs into retrieval documents and structured table rows."""
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

KNOWN_INSURERS = (
    "Star Health",
    "Aditya Birla Health",
    "ICICI Lombard",
    "HDFC ERGO",
    "Bharti AXA",
    "Future Generali",
    "IFFCO Tokio",
    "Navi General",
    "Reliance General",
    "TATA AIG",
    "Universal Sompo",
    "Acko General",
    "New India Assurance",
    "Oriental Insurance",
    "Care Health",
    "Niva Bupa",
    "SBI General",
    "Cholamandalam",
    "Bajaj Allianz",
)

UIN_PREFIX_TO_INSURER = {
    "SHA": "Star Health",
    "ADI": "Aditya Birla Health",
    "ICI": "ICICI Lombard",
    "HDF": "HDFC ERGO",
    "BHA": "Bharti AXA",
    "FGI": "Future Generali",
    "IFF": "IFFCO Tokio",
    "NAV": "Navi General",
    "RHI": "Reliance General",
    "TAT": "TATA AIG",
    "UNI": "Universal Sompo",
    "ACK": "Acko General",
    "NIA": "New India Assurance",
    "OBI": "Oriental Insurance",
    "BAJ": "Bajaj Allianz",
    "CHI": "Cholamandalam",
    "NBH": "Niva Bupa",
}


def extract_uin(pdf_name: str, sample_text: str) -> str:
    """Find a policy UIN in the filename or sampled PDF text."""
    match = UIN_PATTERN.search(pdf_name.upper())
    if match:
        return match.group(0)

    match = UIN_PATTERN.search(sample_text.upper())
    if match:
        return match.group(0)

    match = LEGACY_UIN_PATTERN.search(sample_text.upper())
    if match:
        return match.group(0).rstrip("./-)")

    return ""


def sample_pages_for_uin(pdf, page_limit: int = 4) -> str:
    """Return text from early pages and the final page for UIN lookup."""
    indexes = list(range(min(page_limit, pdf.page_count)))
    if pdf.page_count and (pdf.page_count - 1) not in indexes:
        indexes.append(pdf.page_count - 1)

    return "\n".join(pdf[index].get_text() for index in indexes)


def extract_insurer(first_page_text: str, uin: str) -> str:
    """Identify the insurer from page text or the UIN prefix."""
    lowered = first_page_text.lower()
    for insurer in KNOWN_INSURERS:
        if insurer.lower() in lowered:
            return insurer

    return UIN_PREFIX_TO_INSURER.get(uin[:3].upper(), "Unknown Insurer")


def extract_product(pdf_name: str) -> str:
    """Create a readable product name from the PDF filename."""
    stem = Path(pdf_name).stem
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem)
    return stem.strip()


def _build_document(text: str, metadata: dict, pdf_name: str):
    """Create a LangChain document with stable source metadata."""
    from langchain_core.documents import Document

    full_metadata = {
        **metadata,
        "source": pdf_name,
        "page_number": metadata["page"],
        "category": metadata["chunk_type"],
    }
    return Document(page_content=text, metadata=full_metadata)


def _table_documents(
    table: dict,
    table_type: str,
    insurer: str,
    uin: str,
    product: str,
    page: int,
    pdf_name: str,
) -> list:
    """Convert one table into citable retrieval documents."""
    sentences = table_to_sentences(table, table_type, insurer, uin, page)
    metadata = {
        "uin": uin,
        "insurer": insurer,
        "product": product,
        "page": page,
        "section": table_type,
        "topic": table_type,
        "topics": retrieval_topics(table_type),
        "chunk_type": "table_sentence",
    }
    return [
        _build_document(sentence, metadata, pdf_name)
        for sentence in sentences
    ]


def _send_to_pinecone(
    table: dict,
    table_type: str,
    context: dict,
    counters: dict,
    documents: list,
) -> None:
    """Append retrieval documents created from a table."""
    table_documents = _table_documents(
        table,
        table_type,
        context["insurer"],
        context["uin"],
        context["product"],
        context["page"],
        context["pdf_name"],
    )
    documents.extend(table_documents)
    counters["table_sentences"] += len(table_documents)


def _stage_for_sql(
    table: dict,
    table_type: str,
    context: dict,
    counters: dict,
    staged_rows: list[dict],
) -> None:
    """Append structured rows created from a table."""
    rows = table_to_rows(table, table_type)
    for row in rows:
        staged_rows.append(
            {
                "uin": context["uin"],
                "insurer": context["insurer"],
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
    """Classify a table and route it to retrieval, SQL staging, or both."""
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
    """Chunk page text and route any tables found on the page."""
    clean_text, tables = parse_page(page)
    page_number = context["page"]

    chunks = chunk_page(
        clean_text,
        context["uin"],
        context["insurer"],
        context["product"],
        page_number,
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

    notes = []
    for table in tables:
        note = _route_table(
            table,
            context,
            counters,
            documents,
            staged_rows,
        )
        if note:
            notes.append(note)

    if chunks or notes:
        suffix = f" | tables: {', '.join(notes)}" if notes else ""
        print(f"    page {page_number:>3}: {len(chunks):>2} chunks{suffix}")

    return last_topics(chunks) or context.get("carry_topics")


def process_pdf(pdf_path: Path, counters: dict) -> tuple[list, list[dict]]:
    """Parse one PDF into retrieval documents and structured rows."""
    import pymupdf

    documents: list = []
    staged_rows: list[dict] = []

    with pymupdf.open(pdf_path) as pdf:
        first_page_text = pdf[0].get_text() if pdf.page_count else ""
        uin = extract_uin(pdf_path.name, sample_pages_for_uin(pdf))

        if not uin:
            print(f"\n  {pdf_path.name}")
            print("    SKIPPED: no UIN printed anywhere in this document")
            logger.warning("pdf_skipped_no_uin name=%s", pdf_path.name)
            counters["skipped_no_uin"] = counters.get("skipped_no_uin", 0) + 1
            return [], []

        insurer = extract_insurer(first_page_text, uin)
        product = extract_product(pdf_path.name)

        print(f"\n  {pdf_path.name}")
        print(f"    insurer={insurer} | uin={uin} | pages={pdf.page_count}")

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
    """Return counters for one indexing run."""
    return {
        "pdfs": 0,
        "chunks": 0,
        "table_sentences": 0,
        "sql_rows": 0,
        "tables_skipped": 0,
        "tables_by_type": {},
    }


def save_staged_tables(staged_rows: list[dict]) -> Path:
    """Write structured table rows to the artifacts folder."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = settings.artifacts_dir / "staged_tables.json"
    path.write_text(json.dumps(staged_rows, indent=2), encoding="utf-8")
    return path


def print_summary(counters: dict, staged_path: Path) -> None:
    """Print a summary of the indexing run."""
    print("")
    print("=" * 60)
    print("INDEXING SUMMARY")
    print("=" * 60)
    print(f"  PDFs processed      : {counters['pdfs']}")
    print(f"  Text chunks         : {counters['chunks']}")
    print(f"  Table sentences     : {counters['table_sentences']}")
    print(f"  Rows staged for SQL : {counters['sql_rows']}")
    print(f"  Staged rows file    : {staged_path}")
    print(f"  Tables skipped      : {counters['tables_skipped']}")
    print("  Tables by type:")

    ranked = sorted(
        counters["tables_by_type"].items(),
        key=lambda item: -item[1],
    )
    for table_type, count in ranked:
        print(f"    {table_type:<22} {count}")

    print("=" * 60)


def _collect_documents_from_pdfs(
    pdf_paths: list[Path],
    counters: dict,
) -> tuple[list, list[dict]]:
    """Process all PDFs and collect successful outputs."""
    all_documents: list = []
    all_staged_rows: list[dict] = []

    for pdf_path in pdf_paths:
        try:
            documents, staged_rows = process_pdf(pdf_path, counters)
        except Exception:
            logger.exception("pdf_failed path=%s", pdf_path)
            print(f"    -- failed, skipping {pdf_path.name}")
            continue

        all_documents.extend(documents)
        all_staged_rows.extend(staged_rows)

    return all_documents, all_staged_rows


def run_pipeline(push_to_pinecone: bool = True) -> dict:
    """Process all policy PDFs and optionally index the documents."""
    pdf_paths = sorted(settings.data_dir.glob("*.pdf"))
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
