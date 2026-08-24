"""Decide what to do with each table found in a policy PDF.

WHY: not every table is worth indexing. An ombudsman office address list is
pure noise in a vector index, while a sub-limit table is exactly what a
customer question needs. This module sorts tables into four destinations:

    skip               -- throw it away
    sql_and_pinecone   -- rows into SQLite, sentences into Pinecone
    sql_only           -- rows into SQLite (too long/tabular to embed well)
    pinecone_only      -- sentences into Pinecone (no clean row structure)
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Rupee figures and percentages -- their absence marks a table as a bare
# item list rather than a limits or rates table.
AMOUNT_PATTERN = re.compile(
    r"(rs\.?\s*\d|inr\s*\d|\d+\s*%|\d{1,3}(,\d{2,3})+|\d{4,})", re.I
)

# --- destinations ------------------------------------------------------
SKIP = "skip"
SQL_AND_PINECONE = "sql_and_pinecone"
SQL_ONLY = "sql_only"
PINECONE_ONLY = "pinecone_only"

# --- junk: contact directories and non-payable item lists ----------------
# These are checked first -- an ombudsman table may still mention "limit".
SKIP_KEYWORDS = (
    "ombudsman",
    "bimalokpal",
    "jeevan prakash",
    "jurisdiction of office",
    "baby food",
    "hair removal cream",
    "baby charges",
    "admission/registration charges",
    "policy no",
    "health card no",
    "name as per bank account",
    "ifsc",
    "usgi coins",
    "fitness assessment",
    "active dayz",
    # grievance / contact directories
    "first point of contact",
    "grievance redressal",
    "grievance officer",
    # non-payable item lists, which run for pages under many headings
    "non-payable",
    "non payable",
    "list of non",
)

# --- tables that give us both structured rows and good sentences ---------
SQL_AND_PINECONE_KEYWORDS = {
    "sub_limit": ("sub-limit", "cataract", "hernia", "limit per claim"),
    "waiting_period": ("waiting period", "ailment / disease", "benign ent"),
    "copayment": ("co-payment", "co-pay", "patient shall bear"),
    "room_rent": ("room rent limit", "icu charges", "per day"),
    "plan_comparison": ("silver", "gold", "diamond", "vital plan", "topaz"),
    "accidental_payout": (
        "loss covered",
        "percentage of sum insured",
        "accidental death",
    ),
    "modern_treatment": (
        "robotic",
        "deep brain",
        "oral chemotherapy",
        "stereotactic",
    ),
}

# --- lookup tables: far too many rows to embed, perfect for SQL ----------
SQL_ONLY_KEYWORDS = {
    "premium_rate": ("si/age", "age band", "21-35 yrs", "36-45 yrs"),
    "day_care_procedure": (
        "adenoidectomy",
        "appendectomy",
        "list of day care",
    ),
}

# --- prose-shaped tables: no clean row structure, but useful text --------
PINECONE_ONLY_KEYWORDS = {
    "cancellation_refund": (
        "period on risk",
        "refund of premium",
        "cancellation grid",
        "timing of cancellation",
        "risk is retained",
        "% of premium",
    ),
    "entry_age": ("minimum entry age", "maximum entry age", "entry age"),
    "restoration": ("restoration amount", "restoration of sum insured"),
    "pinecone_only": ("zone a/b", "prescribed time limit", "grace period"),
}


def table_to_text(table: dict) -> str:
    """Flatten every cell in a table into one lowercase string for matching."""
    parts = []
    for row in table.get("rows", []):
        for cell in row:
            if cell:
                parts.append(str(cell))
    return " ".join(parts).lower()


def _matches_any(haystack: str, keywords: tuple[str, ...]) -> bool:
    """Return True if any keyword appears in the flattened table text."""
    return any(keyword in haystack for keyword in keywords)


def _is_hospital_network(table: dict) -> bool:
    """A hospital list is identified by its columns, not by its content."""
    header = " ".join(str(cell or "") for cell in table.get("header", [])).lower()
    return "hospital name" in header and "address" in header


def _is_numbered_item_list(table: dict) -> bool:
    """Detect a bare "1. item / 2. item" list with no amounts attached.

    Insurers publish their non-payable items this way, spread over several
    pages under a dozen different headings, so keyword matching alone keeps
    missing them. The shape is the reliable signal: a serial number column
    and item names, with no rupee figure or percentage anywhere in sight.
    """
    rows = table.get("rows", [])
    if len(rows) < 3:
        return False

    numbered = 0
    for row in rows:
        first = str(row[0] or "").strip().rstrip(".")
        if first.isdigit():
            numbered += 1

    # Most rows must start with a serial number...
    if numbered < len(rows) * 0.7:
        return False

    # ...and the table must quote no money and no percentages.
    text = table_to_text(table)
    return not AMOUNT_PATTERN.search(text)


def classify_table(table: dict) -> dict:
    """Return {'table_type', 'destination'} for one extracted table."""
    text = table_to_text(table)

    # 1. Junk goes first so it can never be rescued by a later keyword match.
    if _matches_any(text, SKIP_KEYWORDS):
        return {"table_type": "junk", "destination": SKIP}

    # 2. Hospital networks are column-shaped, so check them before keywords.
    if _is_hospital_network(table):
        return {"table_type": "hospital_network", "destination": SQL_ONLY}

    for table_type, keywords in SQL_AND_PINECONE_KEYWORDS.items():
        if _matches_any(text, keywords):
            return {"table_type": table_type, "destination": SQL_AND_PINECONE}

    for table_type, keywords in SQL_ONLY_KEYWORDS.items():
        if _matches_any(text, keywords):
            return {"table_type": table_type, "destination": SQL_ONLY}

    for table_type, keywords in PINECONE_ONLY_KEYWORDS.items():
        if _matches_any(text, keywords):
            return {"table_type": table_type, "destination": PINECONE_ONLY}

    # 3. Checked after the named types so a numbered day-care procedure
    #    list still reaches SQL instead of being thrown away here.
    if _is_numbered_item_list(table):
        return {"table_type": "non_payable_list", "destination": SKIP}

    # 4. Anything we cannot name is skipped, but logged so we can improve
    #    the keyword lists later.
    logger.warning("unknown_table_type preview=%s", text[:120])
    return {"table_type": "unknown", "destination": SKIP}
