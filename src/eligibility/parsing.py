"""Parse structured values from retrieved policy text."""
from __future__ import annotations

import re

_RUPEE_PATTERN = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d][\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)
_MONTHS_PATTERN = re.compile(
    r"(\d{1,3})\s*(?:\(|\)|[a-z ]{0,20})?\s*month",
    re.IGNORECASE,
)
_YEARS_PATTERN = re.compile(
    r"(\d{1,2})\s*(?:\(|\)|[a-z ]{0,20})?\s*year",
    re.IGNORECASE,
)
_PERCENT_PATTERN = re.compile(
    r"(\d{1,3}(?:\.\d+)?)\s*(?:%|per\s*cent)",
    re.IGNORECASE,
)


def _to_number(raw: str) -> float | None:
    """Convert a captured number to float."""
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_rupee_amount(text: str) -> float | None:
    """Return the smallest valid rupee amount found in text."""
    amounts = [
        value
        for value in (
            _to_number(match)
            for match in _RUPEE_PATTERN.findall(text or "")
        )
        if value is not None and value >= 1000
    ]
    return min(amounts) if amounts else None


def parse_waiting_period_months(text: str) -> int | None:
    """Return a waiting period in months."""
    body = text or ""

    months = [
        int(match)
        for match in _MONTHS_PATTERN.findall(body)
        if 0 < int(match) <= 120
    ]
    if months:
        return max(months)

    years = [
        int(match)
        for match in _YEARS_PATTERN.findall(body)
        if 0 < int(match) <= 10
    ]
    if years:
        return max(years) * 12

    return None


def parse_percent(text: str) -> float | None:
    """Return the largest valid percentage found in text."""
    percents = [
        value
        for value in (
            _to_number(match)
            for match in _PERCENT_PATTERN.findall(text or "")
        )
        if value is not None and 0 < value <= 100
    ]
    return max(percents) if percents else None


def first_match(hits: list[dict], parser, keyword: str | None = None):
    """Return the first parsed value and its source hit."""
    for hit in hits or []:
        text = hit.get("text") or ""
        if keyword and keyword.lower() not in text.lower():
            continue

        value = parser(text)
        if value is not None:
            return value, hit

    return None, None
