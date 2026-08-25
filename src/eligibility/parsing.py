"""Pull numbers out of policy clause text.

WHY this exists: structured facts (sub-limits, waiting periods, co-pays,
deductibles) are only in SQL for policies someone has already curated. A
customer arriving with any of the other indexed policies has no rows at all,
and the engine used to treat that as "no limit applies" -- which quietly
produced a too-generous payout. These helpers let the engine read the number
straight out of the retrieved clause instead.

Everything here is deliberately conservative: if a value cannot be read with
confidence, the function returns None and the caller falls back to a safe
default rather than guessing.
"""
from __future__ import annotations

import re

# Indian policy wordings write amounts as Rs. 40,000 / ₹1,00,000 / INR 40000/-
_RUPEE_PATTERN = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d][\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

# "24 months", "48 (forty eight) months", "twenty four (24) months"
_MONTHS_PATTERN = re.compile(r"(\d{1,3})\s*(?:\(|\)|[a-z ]{0,20})?\s*month", re.IGNORECASE)
_YEARS_PATTERN = re.compile(r"(\d{1,2})\s*(?:\(|\)|[a-z ]{0,20})?\s*year", re.IGNORECASE)

# "5%", "co-payment of 20 %", "20 per cent"
_PERCENT_PATTERN = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|per\s*cent)", re.IGNORECASE)


def _to_number(raw: str) -> float | None:
    """Turn a captured numeric string into a float, or None if it is junk."""
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_rupee_amount(text: str) -> float | None:
    """Return the smallest rupee amount stated in the text, or None.

    Smallest, because sub-limit clauses are usually written as a choice --
    "25% of sum insured or Rs 40,000 whichever is lower" -- and the lower
    figure is the one that actually caps the claim. Picking the larger one
    would overstate what the insurer pays.
    """
    amounts = [
        value
        for value in (_to_number(match) for match in _RUPEE_PATTERN.findall(text or ""))
        # Sub-limits below a thousand rupees are almost always page numbers or
        # clause references that happened to follow a currency symbol.
        if value is not None and value >= 1000
    ]
    return min(amounts) if amounts else None


def parse_waiting_period_months(text: str) -> int | None:
    """Return a waiting period in months, converting years where needed."""
    body = text or ""

    months = [int(m) for m in _MONTHS_PATTERN.findall(body)]
    # A "24 hours" hospitalisation rule is not a waiting period; month values
    # above 120 (10 years) are not either.
    months = [m for m in months if 0 < m <= 120]
    if months:
        return max(months)

    years = [int(y) for y in _YEARS_PATTERN.findall(body) if 0 < int(y) <= 10]
    if years:
        return max(years) * 12

    return None


def parse_percent(text: str) -> float | None:
    """Return the largest percentage stated in the text, or None."""
    percents = [
        value
        for value in (_to_number(match) for match in _PERCENT_PATTERN.findall(text or ""))
        if value is not None and 0 < value <= 100
    ]
    return max(percents) if percents else None


def first_match(hits: list[dict], parser, keyword: str | None = None):
    """Run a parser over retrieved chunks and return the first value found.

    Pass a keyword to require it in the chunk, so a co-pay figure is not read
    out of a clause that happens to mention some unrelated percentage.
    """
    for hit in hits or []:
        text = hit.get("text") or ""
        if keyword and keyword.lower() not in text.lower():
            continue
        value = parser(text)
        if value is not None:
            return value, hit
    return None, None
