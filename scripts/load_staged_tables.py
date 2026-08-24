"""Report on artifacts/staged_tables.json, and (only if forced) load from it.

Run it with:  uv run python scripts/load_staged_tables.py
It REPORTS by default and writes nothing. Writing needs --write, and you
should read the warning below before passing that flag.

DO NOT --write against the current extraction. Filtering the staged rows down
to the ones with a cleanly parseable number is not enough, because the number
is often real but means something else. Sampled from the rows that pass every
filter here:

    40,000  "Cataract Treatment"                            <- correct
         2  "Sinusitis and related disorders."              <- a YEAR count
         5% "5% co pay on all claims"                       <- a co-pay
        25% "Accidental Hospitalisation - 25% increase ..."  <- an INCREASE
       100% "Restoration of the Sum Insured"                 <- a restore
        10% "Death succeeding a hospitalization claim ..."    <- a DISCOUNT

Loading those would tell the engine a claim is capped at Rs 2 because a
waiting-period table said "2 years", or cap it by a discount clause. The
co-payment rows are worse: "0% | above 500 to 1000" is a room-rent slab
bound, and loading it applies a co-pay that does not exist.

The fix is upstream, in table_classifier/table_converter: the table type has
to be right, and the header row has to be found, before these rows can be
trusted. Until then the skipped rows are still reachable through RAG, where
they carry a citation the employee can check -- which is strictly safer than a
wrong number in the table the engine trusts first.

Scale of the problem, measured over the current 1,333 staged rows:

  * only 13% of the sub_limit rows contain an amount or a percentage at all
  * 30% of all rows have a row index ("i", "ii", "3") as their column header,
    because PyMuPDF picked a data row as the header row
  * of the 53 sub_limit rows that survive every filter here, a sample of 12
    contained roughly 3 genuine sub-limits

The report below counts what *would* load, so the number is a progress
measure for fixing the classifier -- not a suggestion to run --write.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings  # noqa: E402
from src.policy_data import PolicyDataStore  # noqa: E402

STAGED_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "staged_tables.json"

# "Rs. 40,000", "INR 40000". The leading \d matters: [\d,]+ alone matches a
# bare comma, and "Rs. ," then parsed as an empty amount.
AMOUNT_PATTERN = re.compile(r"(?:rs\.?|inr|₹)\s*(\d[\d,]*(?:\.\d+)?)", re.I)
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
MONTHS_PATTERN = re.compile(r"(\d+)\s*month", re.I)
YEARS_PATTERN = re.compile(r"(\d+)\s*year", re.I)

# A header that is just a numeral or a roman numeral means the real header
# was never found, so the column names cannot be trusted.
JUNK_HEADER = re.compile(r"^[ivxlc]+$|^\d+$", re.I)

# Column names that hold the thing being limited rather than the limit.
SUBJECT_HINTS = (
    "ailment", "disease", "surgery", "procedure", "treatment", "benefit",
    "particular", "feature", "cover", "condition", "item", "description",
)


def is_trustworthy(row: dict) -> bool:
    """Reject rows whose header row was mis-detected during extraction."""
    return not any(JUNK_HEADER.match(str(key).strip()) for key in row)


def pick_subject(row: dict) -> str | None:
    """Find the column naming what the limit applies to.

    Prefers a column whose header says so ("Ailment / Disease / Surgery"),
    and otherwise falls back to the longest text value, which in these tables
    is nearly always the description rather than a code or an amount.
    """
    for key, value in row.items():
        if key == "table_type" or not str(value).strip():
            continue
        if any(hint in str(key).lower() for hint in SUBJECT_HINTS):
            return str(value).strip()

    candidates = [
        str(value).strip()
        for key, value in row.items()
        if key != "table_type" and len(str(value).strip()) > 4
        and not AMOUNT_PATTERN.search(str(value))
    ]
    return max(candidates, key=len) if candidates else None


def _row_text(row: dict) -> str:
    """Flatten a row's values into one searchable string."""
    return " ".join(str(v) for k, v in row.items() if k != "table_type")


def parse_amount(row: dict) -> tuple[float, str] | None:
    """Return (value, limit_type) when the row states a real limit."""
    text = _row_text(row)

    match = AMOUNT_PATTERN.search(text)
    if match:
        return float(match.group(1).replace(",", "")), "amount"

    match = PERCENT_PATTERN.search(text)
    if match:
        return float(match.group(1)), "percent_si"
    return None


def parse_months(row: dict) -> int | None:
    """Return a waiting period in months, converting years when needed."""
    text = _row_text(row)

    match = MONTHS_PATTERN.search(text)
    if match:
        return int(match.group(1))

    match = YEARS_PATTERN.search(text)
    if match:
        return int(match.group(1)) * 12
    return None


def load_sub_limits(store, rows, counts, dry_run) -> None:
    """Insert sub-limit rows that actually name an amount or a percentage."""
    for entry in rows:
        row = entry["row"]
        subject = pick_subject(row)
        parsed = parse_amount(row)
        if not (is_trustworthy(row) and subject and parsed):
            counts["sub_limit_skipped"] += 1
            continue

        value, limit_type = parsed
        if not dry_run:
            store.add_sub_limit(
                policy_uin=entry["uin"],
                insurer=entry["insurer"],
                treatment=subject[:200],
                limit_amount=value,
                page=int(entry["page"]),
                limit_type=limit_type,
            )
        counts["sub_limit_loaded"] += 1


def load_waiting_periods(store, rows, counts, dry_run) -> None:
    """Insert waiting-period rows that state a number of months or years."""
    for entry in rows:
        row = entry["row"]
        subject = pick_subject(row)
        months = parse_months(row)
        if not (is_trustworthy(row) and subject and months):
            counts["waiting_period_skipped"] += 1
            continue

        if not dry_run:
            store.add_waiting_period(
                policy_uin=entry["uin"],
                insurer=entry["insurer"],
                condition=subject[:200],
                waiting_period_months=months,
                page=int(entry["page"]),
            )
        counts["waiting_period_loaded"] += 1


def load_copayments(store, rows, counts, dry_run) -> None:
    """Insert co-payment rows that state a percentage."""
    for entry in rows:
        row = entry["row"]
        match = PERCENT_PATTERN.search(_row_text(row))
        if not (is_trustworthy(row) and match):
            counts["copayment_skipped"] += 1
            continue

        if not dry_run:
            store.add_copayment(
                policy_uin=entry["uin"],
                insurer=entry["insurer"],
                condition=(pick_subject(row) or "all claims")[:200],
                copay_percent=float(match.group(1)),
                page=int(entry["page"]),
            )
        counts["copayment_loaded"] += 1


def main() -> None:
    """Load the parseable staged rows, reporting what was skipped and why."""
    # Reporting is the default: the rows are not yet trustworthy enough to
    # feed the engine's SQL path. See the warning at the top of this file.
    dry_run = "--write" not in sys.argv

    if not STAGED_PATH.exists():
        print(f"No staged file at {STAGED_PATH} -- run scripts/reindex.py first.")
        return

    staged = json.loads(STAGED_PATH.read_text(encoding="utf-8"))
    print(f"Read {len(staged)} staged rows from {STAGED_PATH.name}")
    if dry_run:
        print("REPORT ONLY -- nothing written (pass --write to override)")

    by_type: dict[str, list] = {}
    for entry in staged:
        by_type.setdefault(entry.get("table_type", ""), []).append(entry)

    store = PolicyDataStore(settings.crm_db_path)
    counts: dict[str, int] = {
        f"{name}_{state}": 0
        for name in ("sub_limit", "waiting_period", "copayment")
        for state in ("loaded", "skipped")
    }

    load_sub_limits(store, by_type.get("sub_limit", []), counts, dry_run)
    load_waiting_periods(store, by_type.get("waiting_period", []), counts, dry_run)
    load_copayments(store, by_type.get("copayment", []), counts, dry_run)

    print("")
    for name in ("sub_limit", "waiting_period", "copayment"):
        loaded, skipped = counts[f"{name}_loaded"], counts[f"{name}_skipped"]
        total = loaded + skipped
        share = f"{100 * loaded / total:.0f}%" if total else "-"
        print(f"  {name:<16} loaded {loaded:>4} / {total:<4} ({share})")

    print("")
    print("Skipped rows are not lost -- their table sentences are in Pinecone")
    print("with a citation, so the engine still reaches them through RAG.")


if __name__ == "__main__":
    main()
