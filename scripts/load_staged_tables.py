"""Load the trustworthy rows of artifacts/staged_tables.json into crm.db.

Run it with:  uv run python scripts/load_staged_tables.py --write
Without --write it reports what would load and changes nothing.

Sub-limits and waiting periods only. Co-payments are deliberately left to
retrieval -- their tables are benefit summaries where a real age-banded
co-pay shares a table with a maternity sub-limit and a family discount, and
no row-level rule separated them reliably. A wrong co-pay silently reduces a
payout, so it is safer to read that clause with a citation.

The eligibility engine looks a number up in these tables before it falls back
to retrieval, and treats what it finds as authoritative. A wrong row here
silently changes a payout, with no citation for the employee to check, so a
row is only written when it can be shown to mean what its column says.

Three filters do that work, each earned from a row that got through without
it:

  * NOT_A_LIMIT     -- a bonus, a restore or a no-claim discount states a
                       percentage that parses exactly like a cap. "25%
                       increase in balance SI" is not a 25% limit.
  * NOT_A_SUBJECT   -- "66-70" and "1 year" are an age band and a duration
                       that landed in the subject column, not things a limit
                       can apply to.
  * NOT_A_CONDITION -- "Only PEDs declared in the Proposal Form ..." is the
                       definition of a waiting period, not a condition it
                       applies to.

Everything skipped is still reachable through retrieval, where it carries a
citation -- strictly safer than a wrong number in the table the engine
consults first.
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

# "Rs. 40,000", "INR 40000", "Rs 1,00,000".
#
# The digit groups allow a space around the comma because PDF text
# extraction inserts one: a real policy reads "Rs.40, 000/-", and a pattern
# that stopped at the space parsed it as 40 -- a thousandfold understatement
# of a genuine cataract sub-limit. Requiring a comma between groups keeps
# "Rs 5,000 3 times" from being read as 50003.
AMOUNT_PATTERN = re.compile(
    r"(?:rs\.?|inr|₹)\s*(\d{1,3}(?:\s?,\s?\d{2,3})+(?:\.\d+)?|\d+(?:\.\d+)?)",
    re.I,
)

# A sub-limit's subject has to name something. Age bands ("66-70",
# "76 & above") and bare values ("10.0%") arrive when a co-payment-by-age
# table is misread as a limits table, and loading them would cap a claim
# by an age bracket.
NOT_A_SUBJECT = re.compile(
    r"^[\d\s.,%&+/-]*$"           # "10.0%", "66-70"
    r"|^\d+\s*&\s*above$"          # "76 & above"
    r"|^\d+\s*(year|month|day|week)s?$",  # "1 year" -- the value, not a subject
    re.I,
)

# A waiting-period condition names a disease or a procedure. These phrases
# mean the cell holds clause prose or an eligibility rule instead -- "Only
# PEDs declared in the Proposal Form ..." is the definition of the waiting
# period, not a condition it applies to, and loading it would put a 48-month
# wait under a condition no treatment can ever match.
NOT_A_CONDITION = re.compile(
    r"\b(proposal form|policy can be|can be availed|shall be|declared and|"
    r"accepted for coverage|as specified|whichever)\b",
    re.I,
)

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

# A benefit-summary table lists real sub-limits next to things that are not
# limits at all -- a cumulative bonus, a restore, a no-claim discount. The
# percentage in "25% increase in balance SI" parses exactly like the one in
# "25% of sum insured", so the number cannot tell them apart. The wording
# can: anything that ADDS to the cover is not a cap on it.
NOT_A_LIMIT = re.compile(
    r"\b(increase|increment|restor\w*|reinstat\w*|bonus|discount|"
    r"co[- ]?pay\w*|refund|cumulative|add[- ]?on|waiver)\b",
    re.I,
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
        # Strip both separators: extraction leaves "40, 000" for "40,000".
        digits = match.group(1).replace(",", "").replace(" ", "")
        return float(digits), "amount"

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

        # A bonus or a restore is not a cap, however cleanly its number parses,
        # and an age band is not a subject.
        if NOT_A_LIMIT.search(subject) or NOT_A_SUBJECT.match(subject):
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

        if NOT_A_SUBJECT.match(subject) or NOT_A_CONDITION.search(subject):
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
        for name in ("sub_limit", "waiting_period")
        for state in ("loaded", "skipped")
    }

    load_sub_limits(store, by_type.get("sub_limit", []), counts, dry_run)
    load_waiting_periods(store, by_type.get("waiting_period", []), counts, dry_run)

    print("")
    for name in ("sub_limit", "waiting_period"):
        loaded, skipped = counts[f"{name}_loaded"], counts[f"{name}_skipped"]
        total = loaded + skipped
        share = f"{100 * loaded / total:.0f}%" if total else "-"
        print(f"  {name:<16} loaded {loaded:>4} / {total:<4} ({share})")

    print("")
    print("Skipped rows are not lost -- their table sentences are in Pinecone")
    print("with a citation, so the engine still reaches them through RAG.")


if __name__ == "__main__":
    main()
