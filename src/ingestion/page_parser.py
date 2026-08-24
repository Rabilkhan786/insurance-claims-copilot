"""Separate a PDF page's prose from its tables.

WHY this file exists: PyMuPDF's ``page.get_text()`` dumps table cell content
straight into the text stream. That means a sliding-window chunker happily
produces chunks like "Cataract 25000 Hernia 30000 waiting period 24" -- cell
values glued to unrelated prose, which retrieves terribly.

The fix: ask PyMuPDF where the tables are, then drop every text block whose
box sits inside a table box. Prose and tables come back as two clean streams.
"""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

# Text blocks often sit a pixel or two outside the detected table border.
# A small tolerance stops those stragglers leaking into the clean text.
BBOX_TOLERANCE = 5.0


def _boxes_overlap(block_box: tuple, table_box: tuple) -> bool:
    """Return True if a text block box intersects a table box."""
    bx0, by0, bx1, by1 = block_box
    tx0, ty0, tx1, ty1 = table_box

    # Grow the table box slightly so border-hugging blocks still count.
    tx0 -= BBOX_TOLERANCE
    ty0 -= BBOX_TOLERANCE
    tx1 += BBOX_TOLERANCE
    ty1 += BBOX_TOLERANCE

    # Two rectangles miss each other if one is fully left/right/above/below.
    if bx1 < tx0 or bx0 > tx1:
        return False
    if by1 < ty0 or by0 > ty1:
        return False
    return True


def _find_tables(page) -> list[dict]:
    """Return every table on the page as {bbox, rows, header}."""
    try:
        found = page.find_tables()
    except Exception as error:  # PyMuPDF raises on a few malformed pages
        logger.warning("table_detection_failed page=%s error=%s", page.number, error)
        return []

    tables = []
    for table in found.tables:
        rows = table.extract()
        if not rows:
            continue
        tables.append(
            {
                "bbox": tuple(table.bbox),
                "rows": rows,
                "header": list(rows[0]) if rows else [],
                "row_count": len(rows),
                "col_count": max(len(row) for row in rows),
            }
        )
    return tables


def _clean_text_blocks(page, table_boxes: list[tuple]) -> str:
    """Join the page's text blocks, skipping any that fall inside a table."""
    lines = []
    for block in page.get_text("blocks"):
        # A block is (x0, y0, x1, y1, text, block_no, block_type).
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if block[6] != 0:  # block_type 1 means image, not text
            continue

        block_box = (x0, y0, x1, y1)
        if any(_boxes_overlap(block_box, box) for box in table_boxes):
            continue

        text = text.strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


def is_usable_table(table: dict) -> bool:
    """Reject tables too small to carry real policy data."""
    return (
        table["row_count"] >= settings.table_min_rows
        and table["col_count"] >= settings.table_min_cols
    )


def parse_page(page) -> tuple[str, list[dict]]:
    """Split one PyMuPDF page into (clean prose text, list of tables).

    The returned text has zero table cell content in it, so the chunker only
    ever sees real sentences.
    """
    tables = _find_tables(page) if settings.table_extraction_enabled else []
    usable = [table for table in tables if is_usable_table(table)]

    table_boxes = [table["bbox"] for table in tables]
    clean_text = _clean_text_blocks(page, table_boxes)

    return clean_text, usable
