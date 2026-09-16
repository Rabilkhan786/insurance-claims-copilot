"""Convert extracted policy tables into text and structured rows."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

ROW_INDEX_PATTERN = re.compile(r"^\(?(?:[ivxlc]+|\d{1,3})\)?[.)]?$", re.I)
EMPTY_VALUES = {"", "nan", "none", "null", "-", "--", "n/a", "na"}

TEMPLATES = {
    "sub_limit": (
        "Under {insurer} (UIN: {uin}, page {page}), {subject} has a "
        "sub-limit of {value}."
    ),
    "waiting_period": (
        "Under {insurer} (UIN: {uin}, page {page}), waiting period for "
        "{subject} is {value}."
    ),
    "copayment": (
        "Under {insurer} (UIN: {uin}, page {page}), co-payment of {value} "
        "applies for {subject}."
    ),
    "room_rent": (
        "Under {insurer} (UIN: {uin}, page {page}), room rent for SI "
        "{subject} is limited to {value} per day."
    ),
    "plan_comparison": (
        "Under {insurer} (UIN: {uin}, page {page}), {value} plan: {subject} "
        "is {extra}."
    ),
    "accidental_payout": (
        "Under {insurer} (UIN: {uin}, page {page}), {subject} pays out "
        "{value} of sum insured."
    ),
    "modern_treatment": (
        "Under {insurer} (UIN: {uin}, page {page}), {subject} has a coverage "
        "limit of {value}."
    ),
}

GENERIC_TEMPLATE = (
    "Under {insurer} (UIN: {uin}, page {page}), {subject}: {value}."
)


def _clean_cell(cell) -> str:
    """Convert a table cell to a clean single-line string."""
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def _is_empty(value: str) -> bool:
    """Return True when a cell has no useful value."""
    return value.strip().lower() in EMPTY_VALUES


def row_to_sentence(
    row: list,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> str | None:
    """Convert one table row into a retrievable sentence."""
    cells = [_clean_cell(cell) for cell in row]
    if len(cells) < 2:
        return None

    subject, value = cells[0], cells[1]
    extra = cells[2] if len(cells) > 2 else ""

    if _is_empty(subject) or _is_empty(value):
        return None

    template = TEMPLATES.get(table_type, GENERIC_TEMPLATE)
    if table_type == "plan_comparison" and _is_empty(extra):
        template = GENERIC_TEMPLATE

    return template.format(
        insurer=insurer,
        uin=uin,
        page=page,
        subject=subject,
        value=value,
        extra=extra,
    )


def table_to_sentences(
    table: dict,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> list[str]:
    """Convert table data rows into citable sentences."""
    rows = table.get("rows", [])
    if len(rows) < 2:
        return []

    sentences = []
    for row in rows[1:]:
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

    # A continuation table may start with a numbered data row instead of a header.
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
