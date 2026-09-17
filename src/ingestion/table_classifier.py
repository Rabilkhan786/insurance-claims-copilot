"""Classify policy tables without dropping unknown table shapes."""
from __future__ import annotations

import re

SKIP = "skip"
SQL_AND_PINECONE = "sql_and_pinecone"
SQL_ONLY = "sql_only"
PINECONE_ONLY = "pinecone_only"

# These rules describe common insurance concepts, not specific products.
TABLE_RULES = {
    "waiting_period": (
        "waiting period",
        "months waiting",
        "years waiting",
    ),
    "sub_limit": (
        "sub-limit",
        "sublimit",
        "sub limit",
        "limit per claim",
    ),
    "copayment": (
        "co-payment",
        "co-pay",
        "copayment",
        "patient shall bear",
    ),
    "room_rent": (
        "room rent",
        "icu charges",
        "room category",
    ),
    "cancellation_refund": (
        "refund of premium",
        "timing of cancellation",
        "period on risk",
    ),
    "entry_age": (
        "minimum entry age",
        "maximum entry age",
        "entry age",
    ),
    "restoration": (
        "restoration of sum insured",
        "restoration amount",
        "restore sum insured",
    ),
    "premium_rate": (
        "age band",
        "premium rate",
        "premium amount",
    ),
}

TABLE_TYPE_TOPICS = {
    "waiting_period": ["waiting_period"],
    "sub_limit": ["sub_limit", "coverage"],
    "copayment": ["copay", "coverage"],
    "room_rent": ["sub_limit", "coverage"],
    "cancellation_refund": ["claim_procedure"],
    "entry_age": ["definition"],
    "restoration": ["coverage"],
    "premium_rate": ["general"],
    "hospital_network": ["general"],
    "generic_table": ["general"],
}


def retrieval_topics(table_type: str) -> list[str]:
    """Return retrieval topics for a classified table."""
    return TABLE_TYPE_TOPICS.get(table_type, ["general"])


def table_to_text(table: dict) -> str:
    """Flatten table cells into lowercase text for classification."""
    parts = []
    for row in table.get("rows", []):
        for cell in row:
            if cell is not None:
                value = " ".join(str(cell).split())
                if value:
                    parts.append(value)
    return " ".join(parts).lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Return True when any keyword appears in table text."""
    return any(keyword in text for keyword in keywords)


def _is_hospital_network(table: dict) -> bool:
    """Detect a hospital network from common structural columns."""
    header = " ".join(
        str(cell or "")
        for cell in table.get("header", [])
    ).lower()
    return "hospital" in header and "address" in header


def _is_premium_grid(table: dict) -> bool:
    """Detect a premium table from age-style columns and numeric values."""
    header = " ".join(
        str(cell or "")
        for cell in table.get("header", [])
    ).lower()
    has_age = "age" in header or bool(re.search(r"\b\d{2}\s*[-–]\s*\d{2}\b", header))
    has_premium = "premium" in header or "si/age" in header
    return has_age and has_premium


def classify_table(table: dict) -> dict:
    """Return a table type and storage destination.

    Recognized business tables are staged for structured use and indexed for
    retrieval. Unrecognized tables are still indexed as generic RAG content so
    a new PDF never requires a code change just to preserve its information.
    """
    text = table_to_text(table)

    if not text:
        return {"table_type": "empty", "destination": SKIP}

    if _is_hospital_network(table):
        return {
            "table_type": "hospital_network",
            "destination": PINECONE_ONLY,
        }

    if _is_premium_grid(table):
        return {
            "table_type": "premium_rate",
            "destination": SQL_AND_PINECONE,
        }

    for table_type, keywords in TABLE_RULES.items():
        if _contains_any(text, keywords):
            return {
                "table_type": table_type,
                "destination": SQL_AND_PINECONE,
            }

    return {
        "table_type": "generic_table",
        "destination": PINECONE_ONLY,
    }
