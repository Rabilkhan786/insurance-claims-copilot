"""Convert extracted policy tables into retrieval text and structured rows."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

ROW_INDEX_PATTERN = re.compile(r"^\(?(?:[ivxlc]+|\d{1,3})\)?[.)]?$", re.I)
EMPTY_VALUES = {"", "nan", "none", "null", "-", "--", "n/a", "na"}


def _clean_cell(cell) -> str:
    """Convert one table cell to a clean single-line string."""
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def _is_empty(value: str) -> bool:
    """Return True when a cleaned cell has no useful content."""
    return value.strip().lower() in EMPTY_VALUES


def _clean_row(row: list) -> list[str]:
    """Clean a row while preserving its original column positions."""
    return [_clean_cell(cell) for cell in row]


def _source_label(insurer: str, uin: str, page: int) -> str:
    """Build a readable source label from metadata that is actually known."""
    parts = []
    if insurer:
        parts.append(insurer)
    if uin:
        parts.append(f"UIN {uin}")
    parts.append(f"page {page}")
    return ", ".join(parts)


def _generic_row_text(row: list[str]) -> str:
    """Join non-empty row values without inventing labels."""
    values = [value for value in row if not _is_empty(value)]
    return " | ".join(values)


def _header_pairs(header: list[str], row: list[str]) -> list[str]:
    """Pair row values with the correct header positions."""
    pairs = []
    width = max(len(header), len(row))

    for index in range(width):
        value = row[index] if index < len(row) else ""
        if _is_empty(value):
            continue

        name = header[index] if index < len(header) else ""
        name = name if not _is_empty(name) else f"column_{index + 1}"
        pairs.append(f"{name}: {value}")

    return pairs


def row_to_sentence(
    row: list,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> str | None:
    """Convert one unlabeled table row into retrievable text."""
    cleaned = _clean_row(row)
    content = _generic_row_text(cleaned)
    if not content:
        return None

    source = _source_label(insurer, uin, page)
    return f"Policy table ({table_type}; {source}): {content}."


def table_to_sentences(
    table: dict,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> list[str]:
    """Convert table rows into generic, citable retrieval sentences."""
    rows = table.get("rows", [])
    if len(rows) < 2:
        return []

    header = _clean_row(rows[0])
    source = _source_label(insurer, uin, page)
    sentences = []

    for raw_row in rows[1:]:
        row = _clean_row(raw_row)
        pairs = _header_pairs(header, row)

        if pairs:
            content = " | ".join(pairs)
            sentences.append(
                f"Policy table ({table_type}; {source}): {content}."
            )
            continue

        sentence = row_to_sentence(
            raw_row,
            table_type,
            insurer,
            uin,
            page,
        )
        if sentence:
            sentences.append(sentence)

    logger.debug(
        "table_converted type=%s page=%s sentences=%s",
        table_type,
        page,
        len(sentences),
    )
    return sentences


def table_to_rows(table: dict, table_type: str) -> list[dict]:
    """Convert table rows into dictionaries for optional structured storage."""
    rows = table.get("rows", [])
    if len(rows) < 2:
        return []

    first_row = _clean_row(rows[0])
    first_cell = first_row[0] if first_row else ""

    if first_cell and ROW_INDEX_PATTERN.match(first_cell):
        width = max(len(row) for row in rows)
        header = [f"col_{index}" for index in range(width)]
        data_rows = rows
    else:
        header = [
            value if not _is_empty(value) else f"col_{index}"
            for index, value in enumerate(first_row)
        ]
        data_rows = rows[1:]

    records = []
    for raw_row in data_rows:
        row = _clean_row(raw_row)
        if not any(not _is_empty(value) for value in row):
            continue

        record = {}
        for index, value in enumerate(row):
            key = header[index] if index < len(header) else f"col_{index}"
            record[key] = value

        record["table_type"] = table_type
        records.append(record)

    return records
