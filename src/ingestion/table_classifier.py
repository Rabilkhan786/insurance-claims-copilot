"""Classify policy tables and choose where each table should be stored."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

AMOUNT_PATTERN = re.compile(
    r"(rs\.?\s*\d|inr\s*\d|\d+\s*%|\d{1,3}(,\d{2,3})+|\d{4,})",
    re.I,
)

SKIP = "skip"
SQL_AND_PINECONE = "sql_and_pinecone"
SQL_ONLY = "sql_only"
PINECONE_ONLY = "pinecone_only"

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
    "first point of contact",
    "grievance redressal",
    "grievance officer",
    "non-payable",
    "non payable",
    "list of non",
)

SQL_AND_PINECONE_KEYWORDS = {
    "waiting_period": (
        ("waiting period", 6),
        ("ailment / disease", 5),
        ("benign ent", 3),
    ),
    "sub_limit": (
        ("sub-limit", 6),
        ("sublimit", 6),
        ("sub limit", 6),
        ("limit per claim", 5),
        ("cataract", 1),
        ("hernia", 1),
    ),
    "copayment": (
        ("co-payment", 6),
        ("co-pay", 5),
        ("copayment", 6),
        ("patient shall bear", 4),
    ),
    "room_rent": (
        ("room rent limit", 6),
        ("room rent", 4),
        ("icu charges", 5),
        ("per day", 2),
    ),
    "plan_comparison": (
        ("vital plan", 5),
        ("topaz", 5),
        ("silver", 2),
        ("gold", 2),
        ("diamond", 2),
    ),
    "accidental_payout": (
        ("loss covered", 5),
        ("percentage of sum insured", 4),
        ("accidental death", 4),
    ),
    "modern_treatment": (
        ("robotic", 5),
        ("deep brain", 5),
        ("oral chemotherapy", 5),
        ("stereotactic", 5),
    ),
}

MIN_TABLE_SCORE = 3

SQL_ONLY_KEYWORDS = {
    "premium_rate": ("si/age", "age band", "21-35 yrs", "36-45 yrs"),
    "day_care_procedure": (
        "adenoidectomy",
        "appendectomy",
        "list of day care",
    ),
}

PINECONE_ONLY_KEYWORDS = {
    "cancellation_refund": (
        "period on risk",
        "refund of premium",
        "cancellation grid",
        "timing of cancellation",
        "risk is retained",
        "% of premium",
    ),
    "entry_age": (
        "minimum entry age",
        "maximum entry age",
        "entry age",
    ),
    "restoration": (
        "restoration amount",
        "restoration of sum insured",
    ),
    "pinecone_only": (
        "zone a/b",
        "prescribed time limit",
        "grace period",
    ),
}

TABLE_TYPE_TOPICS = {
    "waiting_period": ["waiting_period"],
    "sub_limit": ["sub_limit", "coverage"],
    "copayment": ["copay"],
    "room_rent": ["sub_limit"],
    "plan_comparison": ["coverage"],
    "accidental_payout": ["coverage"],
    "modern_treatment": ["coverage"],
    "restoration": ["coverage"],
    "cancellation_refund": ["claim_procedure"],
    "entry_age": ["definition"],
    "pinecone_only": ["claim_procedure"],
}


def retrieval_topics(table_type: str) -> list[str]:
    """Return retrieval topics for a classified table."""
    return TABLE_TYPE_TOPICS.get(table_type, ["general"])


def table_to_text(table: dict) -> str:
    """Flatten table cells into lowercase text for classification."""
    parts = []
    for row in table.get("rows", []):
        for cell in row:
            if cell:
                parts.append(str(cell))
    return " ".join(parts).lower()


def _matches_any(haystack: str, keywords: tuple[str, ...]) -> bool:
    """Return True when any keyword appears in the table text."""
    return any(keyword in haystack for keyword in keywords)


def _score_type(haystack: str, weighted_keywords: tuple) -> int:
    """Return a weighted score for one table type."""
    return sum(
        weight
        for keyword, weight in weighted_keywords
        if keyword in haystack
    )


def _best_scoring_type(haystack: str) -> tuple[str, int]:
    """Return the highest-scoring structured table type."""
    scores = {
        table_type: _score_type(haystack, keywords)
        for table_type, keywords in SQL_AND_PINECONE_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best, scores[best]


def _is_hospital_network(table: dict) -> bool:
    """Detect a hospital network from its column names."""
    header = " ".join(
        str(cell or "")
        for cell in table.get("header", [])
    ).lower()
    return "hospital name" in header and "address" in header


def _is_numbered_item_list(table: dict) -> bool:
    """Detect numbered item lists that contain no amounts or percentages."""
    rows = table.get("rows", [])
    if len(rows) < 3:
        return False

    numbered = 0
    for row in rows:
        first = str(row[0] or "").strip().rstrip(".")
        if first.isdigit():
            numbered += 1

    if numbered < len(rows) * 0.7:
        return False

    return not AMOUNT_PATTERN.search(table_to_text(table))


def classify_table(table: dict) -> dict:
    """Return a table type and storage destination."""
    text = table_to_text(table)

    if _matches_any(text, SKIP_KEYWORDS):
        return {"table_type": "junk", "destination": SKIP}

    if _is_hospital_network(table):
        return {
            "table_type": "hospital_network",
            "destination": SQL_ONLY,
        }

    best_type, best_score = _best_scoring_type(text)
    if best_score >= MIN_TABLE_SCORE:
        return {
            "table_type": best_type,
            "destination": SQL_AND_PINECONE,
        }

    for table_type, keywords in SQL_ONLY_KEYWORDS.items():
        if _matches_any(text, keywords):
            return {
                "table_type": table_type,
                "destination": SQL_ONLY,
            }

    for table_type, keywords in PINECONE_ONLY_KEYWORDS.items():
        if _matches_any(text, keywords):
            return {
                "table_type": table_type,
                "destination": PINECONE_ONLY,
            }

    if _is_numbered_item_list(table):
        return {
            "table_type": "non_payable_list",
            "destination": SKIP,
        }

    logger.warning("unknown_table_type preview=%s", text[:120])
    return {"table_type": "unknown", "destination": SKIP}
