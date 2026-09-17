"""Classify policy tables without dropping unknown table shapes."""
from __future__ import annotations

import re

SKIP = "skip"
RAG_ONLY = "rag_only"
RAG_AND_SQL = "rag_and_sql"

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
    values = []
    for row in table.get("rows", []):
        for cell in row:
            value = " ".join(str(cell or "").split())
            if value:
                values.append(value)
    return " ".join(values).lower()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _header_text(table: dict) -> str:
    return " ".join(str(cell or "") for cell in table.get("header", [])).lower()


def _is_hospital_network(table: dict) -> bool:
    header = _header_text(table)
    return "hospital" in header and "address" in header


def _is_premium_grid(table: dict) -> bool:
    header = _header_text(table)
    has_age = "age" in header or bool(
        re.search(r"\b\d{2}\s*[-–]\s*\d{2}\b", header)
    )
    has_premium = "premium" in header or "si/age" in header
    return has_age and has_premium


def classify_table(table: dict) -> dict:
    """Return the table type and where it should be stored."""
    text = table_to_text(table)
    if not text:
        return {"table_type": "empty", "destination": SKIP}

    if _is_hospital_network(table):
        return {"table_type": "hospital_network", "destination": RAG_ONLY}

    if _is_premium_grid(table):
        return {"table_type": "premium_rate", "destination": RAG_AND_SQL}

    for table_type, keywords in TABLE_RULES.items():
        if _contains_any(text, keywords):
            return {"table_type": table_type, "destination": RAG_AND_SQL}

    return {"table_type": "generic_table", "destination": RAG_ONLY}
