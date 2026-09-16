"""Chunk policy text at section and sentence boundaries and tag topics."""
from __future__ import annotations

import logging
import re
import unicodedata

from config import settings

logger = logging.getLogger(__name__)

SECTION_PATTERNS = (
    re.compile(r"^\d+\.\d+(\.\d+)*\s+\S"),
    re.compile(r"^\d+\.\s+[A-Z]"),
    re.compile(r"^\([a-z]\)\s+\S"),
    re.compile(r"^Code\s+Excl\.?\s*\d+", re.I),
    re.compile(r"^SECTION\s+[IVXLC]+\b", re.I),
    re.compile(r"^Clause\s+\d+", re.I),
    re.compile(r"^Benefit\s+\d+", re.I),
    re.compile(r"^[A-Z]\.\s+[A-Z]"),
)

SENTENCE_BOUNDARY = re.compile(
    r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)"
    r"(?<!\bi\.e\.)(?<!\be\.g\.)(?<=[.!?])\s+"
)

BOILERPLATE_PATTERNS = (
    re.compile(r"\birdai?\s*(regn|registration)\b", re.I),
    re.compile(r"\b1800[\s\-]?\d{3}[\s\-]?\d{3,4}\b"),
    re.compile(r"\bcin\s*[:\-]", re.I),
    re.compile(r"\bwww\.\S+\.(com|in|org)\b", re.I),
    re.compile(r"^(page\s+)?\d+\s+of\s+\d+$", re.I),
    re.compile(r"^unique\s+identification\s+no", re.I),
    re.compile(r"^[A-Z]{2,5}\s*/\s*[A-Z0-9.]+\s*/", re.I),
    re.compile(r"\buin\s*:.*\b(prospectus|policy word)", re.I),
    re.compile(r"\bemail\s*id\s*:", re.I),
    re.compile(r"\b(ground|\d+(st|nd|rd|th))\s+floor\b", re.I),
)

TOPIC_RULES = (
    (
        "waiting_period",
        (
            (re.compile(r"\bwaiting period\b", re.I), 4),
            (re.compile(r"\bmoratorium\b", re.I), 4),
            (re.compile(r"\bpre[- ]?existing (disease|condition)", re.I), 2),
            (
                re.compile(
                    r"\b\d+\s*(months?|days?|years?)\s+(of\s+)?"
                    r"(continuous\s+)?(coverage|from|after|since)",
                    re.I,
                ),
                2,
            ),
        ),
    ),
    (
        "exclusion",
        (
            (re.compile(r"\bcode[\s\-]*excl", re.I), 5),
            (re.compile(r"\bexpenses related to\b", re.I), 2),
            (
                re.compile(
                    r"\bshall not be (liable|payable|admissible|covered)",
                    re.I,
                ),
                4,
            ),
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
            (
                re.compile(
                    r"\b\d+\s*%\s*(of\s+)?(the\s+)?(admissible|each)\b",
                    re.I,
                ),
                3,
            ),
        ),
    ),
    (
        "sub_limit",
        (
            (re.compile(r"\bsub[- ]?limits?\b", re.I), 5),
            (
                re.compile(
                    r"\b(limited to|up to|maximum of|not exceeding|capped at)\b"
                    r"[^.]{0,25}(rs\.?|inr|₹)\s*[\d,]+",
                    re.I,
                ),
                4,
            ),
            (re.compile(r"\b\d+\s*%\s*of\s+(the\s+)?sum insured\b", re.I), 4),
            (re.compile(r"\broom rent\b", re.I), 3),
            (
                re.compile(
                    r"\bper day\b[^.]{0,30}(rs\.?|inr|₹)",
                    re.I,
                ),
                3,
            ),
        ),
    ),
    (
        "claim_procedure",
        (
            (re.compile(r"\bclaim form\b", re.I), 4),
            (re.compile(r"\bclaim documents\b", re.I), 4),
            (re.compile(r"\bcashless\b", re.I), 2),
            (
                re.compile(
                    r"\bdocuments?\s+(required|to be submitted)\b",
                    re.I,
                ),
                4,
            ),
            (
                re.compile(
                    r"\b(intimat|notif)\w*[^.]{0,40}\b(company|insurer|tpa)\b",
                    re.I,
                ),
                3,
            ),
            (
                re.compile(
                    r"\bwithin\s+\d+\s*(hours?|days?)[^.]{0,40}"
                    r"\b(admission|discharge|intimation)\b",
                    re.I,
                ),
                3,
            ),
            (
                re.compile(
                    r"\b(cashless|reimbursement)\b[^.]{0,30}"
                    r"\b(facility|process|request|claim)\b",
                    re.I,
                ),
                3,
            ),
        ),
    ),
    (
        "coverage",
        (
            (re.compile(r"\bwe will (pay|cover|indemnify|reimburse)\b", re.I), 4),
            (re.compile(r"\b(costs?|expenses) incurred\b", re.I), 3),
            (
                re.compile(
                    r"\b(shall|will)\s+(be\s+)?(indemnif|reimburs)",
                    re.I,
                ),
                4,
            ),
            (re.compile(r"\b(is|are|shall be)\s+(covered|payable)\b", re.I), 4),
            (re.compile(r"\bhospitali[sz]ation expenses\b", re.I), 3),
            (
                re.compile(
                    r"\bbenefits?\s+(is|are|shall be)\s+(payable|available)\b",
                    re.I,
                ),
                3,
            ),
            (re.compile(r"\bin[- ]patient (treatment|care|hospitali)", re.I), 3),
            (re.compile(r"\b(covered|payable|eligible|benefit)\b", re.I), 1),
        ),
    ),
    (
        "definition",
        (
            (re.compile(r"\bis defined as\b", re.I), 3),
            (re.compile(r"\bshall mean\b", re.I), 3),
            (re.compile(r"\brefers to\b", re.I), 3),
            (re.compile(r"^[\"“']?[A-Z][\w \-/]{2,40}[\"”']?\s+means\b"), 4),
            (re.compile(r"\bmeans\b", re.I), 1),
        ),
    ),
)

MIN_TOPIC_SCORE = 3


def score_topics(text: str) -> dict[str, int]:
    """Return weighted topic scores for a chunk."""
    scores = {}
    for topic, rules in TOPIC_RULES:
        total = sum(
            weight
            for pattern, weight in rules
            if pattern.search(text)
        )
        if total:
            scores[topic] = total
    return scores


def detect_topics(text: str) -> list[str]:
    """Return relevant topic tags ordered by score."""
    scores = score_topics(text)
    if not scores:
        return ["general"]

    strong = [
        topic
        for topic, score in scores.items()
        if score >= MIN_TOPIC_SCORE
    ]
    kept = strong or [max(scores, key=scores.get)]
    return sorted(kept, key=lambda topic: -scores[topic])


def is_boilerplate(line: str) -> bool:
    """Return True for repeated headers, contact details, and other noise."""
    lowered = line.lower()
    if any(pattern in lowered for pattern in settings.chunk_skip_patterns):
        return True
    return any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)


def is_section_start(line: str) -> bool:
    """Return True when a line begins a policy section or clause."""
    stripped = line.strip()
    if not stripped:
        return False
    return any(pattern.match(stripped) for pattern in SECTION_PATTERNS)


def normalize_text(text: str) -> str:
    """Normalize common PDF typography before matching and chunking."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return text.replace("\xa0", " ")


def _strip_boilerplate(text: str) -> list[str]:
    """Return non-empty policy text lines without boilerplate."""
    kept = []
    for line in normalize_text(text).splitlines():
        stripped = line.strip()
        if not stripped or is_boilerplate(stripped):
            continue
        kept.append(stripped)
    return kept


def split_into_sections(text: str) -> list[dict]:
    """Group page text under detected policy section headings."""
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
        {
            "section": item["section"],
            "text": " ".join(item["lines"]),
        }
        for item in sections
    ]


def split_long_text(text: str, max_size: int) -> list[str]:
    """Split long text at sentence boundaries."""
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
    """Merge very short pieces with nearby text."""
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

    if carry:
        if merged:
            merged[-1]["text"] = (
                f"{merged[-1]['text']} {carry['text']}".strip()
            )
        else:
            merged.append(carry)

    return merged


LIST_ITEM_START = re.compile(
    r"^\s*(\(?[ivxlc]+[).]|\(?[a-z][).]|\d+[).])\s+",
    re.I,
)


def inherit_list_topics(
    records: list[dict],
    carried: list[str] | None,
) -> list[str] | None:
    """Carry the parent topic into unlabelled list items."""
    for record in records:
        topics = record["metadata"]["topics"]
        if topics != ["general"]:
            carried = topics
        elif carried and LIST_ITEM_START.match(record["text"]):
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
    """Chunk one page and attach policy metadata and topic tags."""
    pieces: list[dict] = []
    for section in split_into_sections(text):
        for piece in split_long_text(
            section["text"],
            settings.max_chunk_size,
        ):
            pieces.append(
                {
                    "section": section["section"],
                    "text": piece,
                }
            )

    records = []
    for piece in merge_short_pieces(pieces, settings.min_chunk_size):
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
    """Return the last strong topic so lists can continue across pages."""
    for record in reversed(records):
        topics = record["metadata"]["topics"]
        if topics != ["general"]:
            return topics
    return None
