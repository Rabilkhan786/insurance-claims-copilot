"""Extract prose and tables from one PDF page."""
from __future__ import annotations

import logging

from config import settings

logger = logging.getLogger(__name__)

BBOX_TOLERANCE = 5.0


def _boxes_overlap(block_box: tuple, table_box: tuple) -> bool:
    """Return True when a text block overlaps a table area."""
    bx0, by0, bx1, by1 = block_box
    tx0, ty0, tx1, ty1 = table_box

    tx0 -= BBOX_TOLERANCE
    ty0 -= BBOX_TOLERANCE
    tx1 += BBOX_TOLERANCE
    ty1 += BBOX_TOLERANCE

    return not (
        bx1 < tx0
        or bx0 > tx1
        or by1 < ty0
        or by0 > ty1
    )


def _extract_tables(page) -> list[dict]:
    """Extract detected tables into a small, consistent dictionary shape."""
    try:
        finder = page.find_tables()
    except Exception as error:
        logger.warning(
            "table_detection_failed page=%s error=%s",
            page.number,
            error,
        )
        return []

    tables = []
    for table in finder.tables:
        rows = table.extract() or []
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


def is_usable_table(table: dict) -> bool:
    """Return True when a detected table is large enough to process."""
    return (
        table.get("row_count", 0) >= settings.table_min_rows
        and table.get("col_count", 0) >= settings.table_min_cols
    )


def _extract_prose(page, excluded_boxes: list[tuple]) -> str:
    """Return page text that is not part of a processed table."""
    lines = []

    for block in page.get_text("blocks"):
        if len(block) < 7 or block[6] != 0:
            continue

        block_box = tuple(block[:4])
        if any(
            _boxes_overlap(block_box, table_box)
            for table_box in excluded_boxes
        ):
            continue

        text = str(block[4]).strip()
        if text:
            lines.append(text)

    return "\n".join(lines)


def parse_page(page) -> tuple[str, list[dict]]:
    """Return page prose plus tables that are suitable for ingestion."""
    if not settings.table_extraction_enabled:
        return page.get_text().strip(), []

    detected_tables = _extract_tables(page)
    usable_tables = [
        table for table in detected_tables if is_usable_table(table)
    ]

    # Only remove tables that we actually keep. Small or unusable detected
    # tables remain in prose so their text is not silently lost.
    table_boxes = [table["bbox"] for table in usable_tables]
    prose = _extract_prose(page, table_boxes)

    return prose, usable_tables
