"""Separate PDF page text from extracted tables."""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

BBOX_TOLERANCE = 5.0


def _boxes_overlap(block_box: tuple, table_box: tuple) -> bool:
    """Return True when a text block intersects a table area."""
    bx0, by0, bx1, by1 = block_box
    tx0, ty0, tx1, ty1 = table_box

    tx0 -= BBOX_TOLERANCE
    ty0 -= BBOX_TOLERANCE
    tx1 += BBOX_TOLERANCE
    ty1 += BBOX_TOLERANCE

    if bx1 < tx0 or bx0 > tx1:
        return False
    if by1 < ty0 or by0 > ty1:
        return False
    return True


def _find_tables(page) -> list[dict]:
    """Extract tables and basic table metadata from one page."""
    try:
        found = page.find_tables()
    except Exception as error:
        logger.warning(
            "table_detection_failed page=%s error=%s",
            page.number,
            error,
        )
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
                "header": list(rows[0]),
                "row_count": len(rows),
                "col_count": max(len(row) for row in rows),
            }
        )
    return tables


def _clean_text_blocks(page, table_boxes: list[tuple]) -> str:
    """Return page text without content that belongs to tables."""
    lines = []

    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]

        if block[6] != 0:
            continue

        block_box = (x0, y0, x1, y1)
        if any(_boxes_overlap(block_box, box) for box in table_boxes):
            continue

        text = text.strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


def is_usable_table(table: dict) -> bool:
    """Return True when a table meets the configured size limits."""
    return (
        table["row_count"] >= settings.table_min_rows
        and table["col_count"] >= settings.table_min_cols
    )


def parse_page(page) -> tuple[str, list[dict]]:
    """Return clean prose and usable tables from one PDF page."""
    tables = _find_tables(page) if settings.table_extraction_enabled else []
    usable_tables = [table for table in tables if is_usable_table(table)]
    table_boxes = [table["bbox"] for table in tables]
    clean_text = _clean_text_blocks(page, table_boxes)

    return clean_text, usable_tables
