"""Split clean page text into chunks at real section and clause boundaries.

WHY: the old sliding-window chunker cut wherever the character budget ran
out, which regularly sliced a clause in half. Half an exclusion clause is
worse than useless -- it retrieves confidently and answers wrongly.

This chunker cuts only where the policy document itself cuts: at numbered
sections, lettered sub-clauses, IRDAI exclusion codes and clause headings.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from config import settings

logger = logging.getLogger(__name__)

# --- where a new section legitimately starts ----------------------------
SECTION_PATTERNS = (
    re.compile(r"^\d+\.\d+(\.\d+)*\s+\S"),        # 4.1 Pre-existing diseases
    re.compile(r"^\d+\.\s+[A-Z]"),                # 4. EXCLUSIONS
    re.compile(r"^\([a-z]\)\s+\S"),               # (a) dental treatment
    re.compile(r"^Code\s+Excl\.?\s*\d+", re.I),   # Code Excl.01
    re.compile(r"^SECTION\s+[IVXLC]+\b", re.I),   # SECTION IV
    re.compile(r"^Clause\s+\d+", re.I),           # Clause 3
    re.compile(r"^Benefit\s+\d+", re.I),          # Benefit 15
    re.compile(r"^[A-Z]\.\s+[A-Z]"),              # A. PREAMBLE
)

# Split only after real sentence ends. The lookbehinds stop "Rs." and
# friends from being mistaken for the end of a sentence.
SENTENCE_BOUNDARY = re.compile(
    r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)"
    r"(?<!\bi\.e\.)(?<!\be\.g\.)(?<=[.!?])\s+"
)

# --- boilerplate that appears on nearly every page ----------------------
BOILERPLATE_PATTERNS = (
    re.compile(r"\birdai?\s*(regn|registration)\b", re.I),
    re.compile(r"\b1800[\s\-]?\d{3}[\s\-]?\d{3,4}\b"),   # toll free numbers
    re.compile(r"\bcin\s*[:\-]", re.I),
    re.compile(r"\bwww\.\S+\.(com|in|org)\b", re.I),
    re.compile(r"^(page\s+)?\d+\s+of\s+\d+$", re.I),      # "5 of 10" header
    re.compile(r"^unique\s+identification\s+no", re.I),   # UIN header line
    re.compile(r"^[A-Z]{2,5}\s*/\s*[A-Z0-9.]+\s*/", re.I),  # POL / ASP / V.4
    # Repeated page headers: "12 The Oriental Insurance Company Ltd. Happy
    # Family Floater Policy-2015 UIN: ... Prospectus". A real clause that
    # cites a UIN never also says "Prospectus" or "Policy Wordings", so
    # pairing the two is a safe way to spot the running header.
    re.compile(r"\buin\s*:.*\b(prospectus|policy word)", re.I),
    re.compile(r"\bemail\s*id\s*:", re.I),
    re.compile(r"\b(ground|\d+(st|nd|rd|th))\s+floor\b", re.I),  # office address
)

# --- topic tags ---------------------------------------------------------
# Each rule is (pattern, weight). A chunk's score for a topic is the sum of
# the weights of the patterns that matched it.
#
# WHY scoring instead of the old first-match-wins list: the old version
# returned the first topic whose keyword appeared anywhere in the chunk, and
# its claim_procedure rule matched the bare word "claim". Almost every clause
# in a policy says "claim", so real coverage clauses came back tagged
# claim_procedure -- and because coverage was checked last, it almost never
# won. Retrieval filters on this tag *before* it searches, so a wrongly
# tagged chunk is invisible to the tool that needed it.
TOPIC_RULES = (
    (
        "waiting_period",
        (
            (re.compile(r"\bwaiting period\b", re.I), 4),
            (re.compile(r"\bmoratorium\b", re.I), 4),
            (re.compile(r"\bpre[- ]?existing (disease|condition)", re.I), 2),
            (re.compile(r"\b\d+\s*(months?|days?|years?)\s+(of\s+)?"
                        r"(continuous\s+)?(coverage|from|after|since)", re.I), 2),
        ),
    ),
    (
        "exclusion",
        (
            # IRDAI writes these as "Code- Excl05", "Code Excl05" and
            # "(Code-Excl05)". Requiring a space missed every hyphenated one,
            # which is most of them -- these are the standard exclusion codes.
            (re.compile(r"\bcode[\s\-]*excl", re.I), 5),
            # Bare "Expenses related to ..." is how the numbered exclusion
            # lists phrase almost every entry.
            (re.compile(r"\bexpenses related to\b", re.I), 2),
            # "shall not be liable" is an exclusion; a bare "shall not" is not
            # -- "the time limit shall not apply" is a scope note.
            (re.compile(r"\bshall not be (liable|payable|admissible|covered)", re.I), 4),
            (re.compile(r"\bshall not (indemnify|cover|pay)\b", re.I), 4),
            (re.compile(r"\bwe will not pay\b", re.I), 4),
            (re.compile(r"\b(permanent )?exclusions?\b", re.I), 3),
            (re.compile(r"\b(not covered|not payable|not admissible)\b", re.I), 3),
            (re.compile(r"\bexcluded\b", re.I), 2),
        ),
    ),
    (
        "copay",
        (
            (re.compile(r"\bco[- ]?pay(ment)?\b", re.I), 5),
            (re.compile(r"\b(insured|you)\s+shall\s+bear\b", re.I), 3),
            (re.compile(r"\b\d+\s*%\s*(of\s+)?(the\s+)?(admissible|each)\b", re.I), 3),
        ),
    ),
    (
        "sub_limit",
        (
            (re.compile(r"\bsub[- ]?limits?\b", re.I), 5),
            # A real sub-limit names an amount. Plain "maximum" used to tag
            # lines like "Maximum Entry Age is 65 years" as a sub-limit.
            (re.compile(r"\b(limited to|up to|maximum of|not exceeding|capped at)\b"
                        r"[^.]{0,25}(rs\.?|inr|₹)\s*[\d,]+", re.I), 4),
            (re.compile(r"\b\d+\s*%\s*of\s+(the\s+)?sum insured\b", re.I), 4),
            (re.compile(r"\broom rent\b", re.I), 3),
            (re.compile(r"\bper day\b[^.]{0,30}(rs\.?|inr|₹)", re.I), 3),
        ),
    ),
    (
        "claim_procedure",
        (
            (re.compile(r"\bclaim form\b", re.I), 4),
            (re.compile(r"\bclaim documents\b", re.I), 4),
            (re.compile(r"\bcashless\b", re.I), 2),
            (re.compile(r"\bdocuments?\s+(required|to be submitted)\b", re.I), 4),
            (re.compile(r"\b(intimat|notif)\w*[^.]{0,40}\b(company|insurer|tpa)\b", re.I), 3),
            (re.compile(r"\bwithin\s+\d+\s*(hours?|days?)[^.]{0,40}"
                        r"\b(admission|discharge|intimation)\b", re.I), 3),
            (re.compile(r"\b(cashless|reimbursement)\b[^.]{0,30}"
                        r"\b(facility|process|request|claim)\b", re.I), 3),
        ),
    ),
    (
        "coverage",
        (
            (re.compile(r"\bwe will (pay|cover|indemnify|reimburse)\b", re.I), 4),
            # Benefit clauses usually promise to meet a cost rather than to
            # "cover" anything: "the costs incurred on transportation of ...".
            (re.compile(r"\b(costs?|expenses) incurred\b", re.I), 3),
            (re.compile(r"\b(shall|will)\s+(be\s+)?(indemnif|reimburs)", re.I), 4),
            (re.compile(r"\b(is|are|shall be)\s+(covered|payable)\b", re.I), 4),
            (re.compile(r"\bhospitali[sz]ation expenses\b", re.I), 3),
            (re.compile(r"\bbenefits?\s+(is|are|shall be)\s+(payable|available)\b", re.I), 3),
            (re.compile(r"\bin[- ]patient (treatment|care|hospitali)", re.I), 3),
            (re.compile(r"\b(covered|payable|eligible|benefit)\b", re.I), 1),
        ),
    ),
    (
        "definition",
        (
            (re.compile(r"\bis defined as\b", re.I), 3),
            (re.compile(r"\bshall mean\b", re.I), 3),
            # "AYUSH Treatment refers to ..." -- the glossary sections and the
            # critical-illness lists are written this way throughout.
            (re.compile(r"\brefers to\b", re.I), 3),
            (re.compile(r"^[\"“']?[A-Z][\w \-/]{2,40}[\"”']?\s+means\b"), 4),
            (re.compile(r"\bmeans\b", re.I), 1),
        ),
    ),
)

# How much evidence a topic needs before it is worth tagging. Below this we
# still keep the single best guess, so a chunk ends up "general" only when
# nothing matched at all.
MIN_TOPIC_SCORE = 3


def score_topics(text: str) -> dict[str, int]:
    """Add up the rule weights that fire for each topic."""
    scores = {}
    for topic, rules in TOPIC_RULES:
        total = sum(weight for pattern, weight in rules if pattern.search(text))
        if total:
            scores[topic] = total
    return scores


def detect_topics(text: str) -> list[str]:
    """Tag a chunk with every topic it has real evidence for, best first.

    A clause is often two things at once -- "cataract is covered up to
    Rs 40,000" is both coverage and a sub-limit -- and tagging it with only
    one of them hides it from the other tool's metadata filter.
    """
    scores = score_topics(text)
    if not scores:
        return ["general"]

    strong = [topic for topic, score in scores.items() if score >= MIN_TOPIC_SCORE]
    # Weak evidence still beats "general", so fall back to the best guess.
    kept = strong or [max(scores, key=scores.get)]
    return sorted(kept, key=lambda topic: -scores[topic])


def is_boilerplate(line: str) -> bool:
    """Return True for office addresses, registration numbers and hotlines."""
    lowered = line.lower()
    if any(pattern in lowered for pattern in settings.chunk_skip_patterns):
        return True
    return any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)


def is_section_start(line: str) -> bool:
    """Return True if this line begins a new numbered or lettered section."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in SECTION_PATTERNS)


def normalize_text(text: str) -> str:
    """Flatten PDF typography into plain ASCII-ish text.

    Policy PDFs are full of ligatures ("beneﬁt"), curly quotes and
    non-breaking spaces. Left alone they break keyword matching and make
    chunks unreadable in the console.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return text.replace("\xa0", " ")


def _strip_boilerplate(text: str) -> list[str]:
    """Drop boilerplate lines and blank lines, keep the rest in order."""
    kept = []
    for line in normalize_text(text).splitlines():
        stripped = line.strip()
        if not stripped or is_boilerplate(stripped):
            continue
        kept.append(stripped)
    return kept


def split_into_sections(text: str) -> list[dict]:
    """Group lines into sections, starting a new one at each heading."""
    lines = _strip_boilerplate(text)
    if not lines:
        return []

    sections: list[dict] = []
    current = {"section": "", "lines": []}

    for line in lines:
        if is_section_start(line) and current["lines"]:
            sections.append(current)
            current = {"section": line[:80], "lines": [line]}
        elif is_section_start(line):
            current = {"section": line[:80], "lines": [line]}
        else:
            current["lines"].append(line)

    if current["lines"]:
        sections.append(current)

    return [
        {"section": item["section"], "text": " ".join(item["lines"])}
        for item in sections
    ]


def split_long_text(text: str, max_size: int) -> list[str]:
    """Break text over max_size into pieces that end on a full stop.

    A single sentence longer than max_size is left whole on purpose --
    never cutting mid-sentence matters more than the size budget.
    """
    if len(text) <= max_size:
        return [text]

    pieces: list[str] = []
    current = ""

    for sentence in SENTENCE_BOUNDARY.split(text):
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_size:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)
    return pieces


def merge_short_pieces(pieces: list[dict], min_size: int) -> list[dict]:
    """Merge short pieces forward, across section boundaries if needed.

    Policy PDFs are full of one-line sub-clauses like "(c) war and warlike
    operations". Merging happens across sections too, otherwise every such
    clause survives as its own useless 30-character chunk. The merged piece
    keeps the section label of the first piece it came from.
    """
    merged: list[dict] = []
    carry: dict | None = None

    for piece in pieces:
        if carry:
            candidate = {
                "section": carry["section"],
                "text": f"{carry['text']} {piece['text']}".strip(),
            }
        else:
            candidate = dict(piece)

        if len(candidate["text"]) < min_size:
            carry = candidate
            continue

        merged.append(candidate)
        carry = None

    # A short tail has no "next" piece, so it joins the previous one.
    if carry:
        if merged:
            merged[-1]["text"] = f"{merged[-1]['text']} {carry['text']}".strip()
        else:
            merged.append(carry)

    return merged


# A list item under a parent clause, e.g. "28. Expenses for organ donor
# screening" or "xxxiv) Medical treatment whilst racing".
LIST_ITEM_START = re.compile(r"^\s*(\(?[ivxlc]+[).]|\(?[a-z][).]|\d+[).])\s+", re.I)


def inherit_list_topics(records: list[dict], carried: list[str] | None) -> list[str] | None:
    """Give an unlabelled list item the topic of the clause it sits under.

    WHY: exclusion lists name the exclusion once in the parent clause ("the
    Company shall not be liable for:") and then just enumerate items. Each
    item is chunked on its own, so it keeps none of the words that made the
    parent an exclusion and used to land in "general" -- unreachable by
    check_exclusion. Organ-donor and adventure-sport exclusions were both
    being lost this way. Returns the topic to carry into the next page,
    because these lists routinely run across a page break.
    """
    for record in records:
        topics = record["metadata"]["topics"]
        if topics != ["general"]:
            carried = topics
        elif carried and LIST_ITEM_START.match(record["text"]):
            # Inherit only the parent's STRONGEST topic, not its whole list.
            # A specific-waiting-period clause scores both waiting_period and
            # exclusion ("shall not be covered for 24 months"). Passing both
            # down made every condition in the list look excluded, so a
            # cataract claim was rejected outright instead of being checked
            # against its 24-month waiting period.
            record["metadata"]["topics"] = [carried[0]]
            record["metadata"]["topic"] = carried[0]
    return carried


def chunk_page(
    text: str,
    uin: str,
    insurer: str,
    product: str,
    page: int,
    carried_topics: list[str] | None = None,
) -> list[dict]:
    """Chunk one page of clean prose into metadata-tagged records."""
    pieces: list[dict] = []
    for section in split_into_sections(text):
        for piece in split_long_text(section["text"], settings.max_chunk_size):
            pieces.append({"section": section["section"], "text": piece})

    records = []
    for piece in merge_short_pieces(pieces, settings.min_chunk_size):
        # Score the heading together with the body: a heading like
        # "4. EXCLUSIONS" is the strongest clue the body has.
        topics = (
            detect_topics(f"{piece['section']} {piece['text']}")
            if settings.tag_topics
            else ["general"]
        )
        records.append(
            {
                "text": piece["text"],
                "metadata": {
                    "uin": uin,
                    "insurer": insurer,
                    "product": product,
                    "page": page,
                    "section": piece["section"],
                    # topic stays a single string so citations and the older
                    # callers keep working; topics is what retrieval filters on.
                    "topic": topics[0],
                    "topics": topics,
                    "chunk_type": "text",
                },
            }
        )

    inherit_list_topics(records, carried_topics)
    logger.debug("page_chunked page=%s chunks=%s", page, len(records))
    return records


def last_topics(records: list[dict]) -> list[str] | None:
    """Topic to carry into the next page, so cross-page lists keep context."""
    for record in reversed(records):
        topics = record["metadata"]["topics"]
        if topics != ["general"]:
            return topics
    return None
