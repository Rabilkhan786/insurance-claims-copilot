"""Turn classified table rows into natural-language sentences for Pinecone.

WHY: an embedding model cannot make sense of a bare row like
``["Cataract", "Rs 25,000"]``. Rewritten as a full sentence that names the
insurer, the UIN and the page, the same row becomes a retrievable fact with
a citation baked in.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# "i", "(ii)", "3.", "12)" -- a serial number, never a column name.
ROW_INDEX_PATTERN = re.compile(r"^\(?(?:[ivxlc]+|\d{1,3})\)?[.)]?$", re.I)

# Cell values that mean "this cell is empty" once a PDF has been parsed.
EMPTY_VALUES = {"", "nan", "none", "null", "-", "--", "n/a", "na"}

# One template per table type. {subject} is the first column, {value} the
# second, {extra} the third when the table has one.
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
    """Normalise one cell into a single-line string."""
    if cell is None:
        return ""
    return " ".join(str(cell).split())


def _is_empty(value: str) -> bool:
    """Return True when a cell holds no usable information."""
    return value.strip().lower() in EMPTY_VALUES


def row_to_sentence(
    row: list,
    table_type: str,
    insurer: str,
    uin: str,
    page: int,
) -> str | None:
    """Render one table row as a sentence, or None if the row is empty."""
    cells = [_clean_cell(cell) for cell in row]
    if len(cells) < 2:
        return None

    subject, value = cells[0], cells[1]
    extra = cells[2] if len(cells) > 2 else ""

    # A row with no subject or no value tells the reader nothing.
    if _is_empty(subject) or _is_empty(value):
        return None

    template = TEMPLATES.get(table_type, GENERIC_TEMPLATE)

    # plan_comparison is the one template that needs a third column.
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
    """Convert every data row of a table into a citable sentence."""
    rows = table.get("rows", [])
    if len(rows) < 2:
        return []

    sentences = []
    for row in rows[1:]:  # row 0 is the header
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
    """Return the table's data rows keyed by header, for SQL storage.

    Prompt 2 owns the actual SQL tables -- this just hands over clean,
    header-labelled rows so nothing has to re-parse the PDF later.
    """
    rows = table.get("rows", [])
    if len(rows) < 2:
        return []

    first = [_clean_cell(cell) for cell in rows[0]]

    # A table split across two pages carries its header only on the first
    # page, so the continuation starts on a data row -- "vi", "3.", "(ii)".
    # Using that as the header produced columns named after a serial number
    # and a paragraph of policy text, and silently swallowed the row itself.
    # Positional names are honest about the column being unknown, and the
    # row survives as data.
    if first and ROW_INDEX_PATTERN.match(first[0]):
        header = [f"col_{index}" for index in range(len(first))]
        data_rows = rows
    else:
        header = [cell or f"col_{index}" for index, cell in enumerate(first)]
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
