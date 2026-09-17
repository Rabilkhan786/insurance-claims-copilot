"""Convert extracted policy tables into retrievable text and structured rows."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

ROW_INDEX_PATTERN = re.compile(r"^\(?(?:[ivxlc]+|\d{1,3})\)?[.)]?$", re.I)
EMPTY_VALUES = {"", "nan", "none", "null", "-", "--", "n/a", "na"}


def _clean_cell(cell) -> str:
    """Convert a table cell to a clean single-line string."""
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def _is_empty(value: str) -> bool:
    """Return True when a cell has no useful value."""
    return value.strip().lower() in EMPTY_VALUES


def _source_label(insurer: str, uin: str, page: int) -> str:
    """Build a readable citation label from available metadata."""
    parts = []
    if insurer:
        parts.append(insurer)
    if uin:
        parts.append(f"UIN {uin}")
    parts.append(f"page {page}")
    return ", ".join(parts)


def _row_values(row: list) -> list[str]:
    """Return non-empty cleaned cells from one table row."""
    return [
        value
        for value in (_clean_cell(cell) for cell in row)
        if not _is_empty(value)
    ]


def row_to_sentence(
    row: list,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> str | None:
    """Convert one table row into a generic retrievable sentence."""
    values = _row_values(row)
    if len(values) < 2:
        return None

    source = _source_label(insurer, uin, page)
    content = " | ".join(values)
    return f"Policy table ({table_type}; {source}): {content}."


def table_to_sentences(
    table: dict,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> list[str]:
    """Convert table rows into citable retrieval sentences."""
    rows = table.get("rows", [])
    if not rows:
        return []

    header = _row_values(rows[0])
    sentences = []

    # Include the header in each sentence when possible. This preserves meaning
    # for new table layouts without needing a custom template for every PDF.
    for row in rows[1:]:
        values = _row_values(row)
        if not values:
            continue

        source = _source_label(insurer, uin, page)
        if header and len(header) == len(values):
            pairs = [f"{name}: {value}" for name, value in zip(header, values)]
            content = " | ".join(pairs)
            sentence = f"Policy table ({table_type}; {source}): {content}."
        else:
            sentence = row_to_sentence(row, table_type, insurer, uin, page)

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
    """Convert table data into header-labelled structured rows."""
    rows = table.get("rows", [])
    if len(rows) < 2:
        return []

    first_row = [_clean_cell(cell) for cell in rows[0]]

    if first_row and ROW_INDEX_PATTERN.match(first_row[0]):
        header = [f"col_{index}" for index in range(len(first_row))]
        data_rows = rows
    else:
        header = [
            cell or f"col_{index}"
            for index, cell in enumerate(first_row)
        ]
        data_rows = rows[1:]

    records = []
    for row in data_rows:
        cells = [_clean_cell(cell) for cell in row]
        if all(_is_empty(cell) for cell in cells):
            continue

        record = dict(zip(header, cells))
        record["table_type"] = table_type
        records.append(record)

    return records
