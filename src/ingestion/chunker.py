"""Clean and chunk policy text for retrieval."""
from __future__ import annotations

import logging
import re
import unicodedata

from config import settings

logger = logging.getLogger(__name__)

SECTION_PATTERNS = (
    re.compile(r"^\d+(?:\.\d+)*[.)]?\s+\S"),
    re.compile(r"^SECTION\s+[A-Z0-9IVXLC]+\b", re.I),
    re.compile(r"^(CLAUSE|BENEFIT|PART)\s+[A-Z0-9IVXLC]+\b", re.I),
    re.compile(r"^[A-Z][.)]\s+[A-Z]"),
)

SENTENCE_BOUNDARY = re.compile(
    r"(?<!\bRs\.)(?<!\bNo\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMrs\.)"
    r"(?<!\bi\.e\.)(?<!\be\.g\.)(?<=[.!?])\s+"
)

BOILERPLATE_PATTERNS = (
    re.compile(r"^(page\s+)?\d+\s+of\s+\d+$", re.I),
    re.compile(r"\bwww\.\S+", re.I),
    re.compile(r"\bemail\s*(id)?\s*:", re.I),
    re.compile(r"\btoll[- ]?free\b", re.I),
    re.compile(r"\b1800[\s-]?\d{3}[\s-]?\d{3,4}\b"),
    re.compile(r"\bunique identification (number|no)\b", re.I),
)

TOPIC_PATTERNS = {
    "waiting_period": (
        re.compile(r"\bwaiting period\b", re.I),
        re.compile(r"\bpre[- ]?existing (disease|condition)\b", re.I),
        re.compile(r"\bmoratorium\b", re.I),
    ),
    "exclusion": (
        re.compile(r"\bexclusions?\b", re.I),
        re.compile(r"\bnot covered\b", re.I),
        re.compile(r"\bnot payable\b", re.I),
        re.compile(r"\bshall not be (liable|payable|covered|admissible)\b", re.I),
        re.compile(r"\bwe will not pay\b", re.I),
    ),
    "copay": (
        re.compile(r"\bco[- ]?pay(?:ment)?\b", re.I),
        re.compile(r"\binsured\s+shall\s+bear\b", re.I),
    ),
    "sub_limit": (
        re.compile(r"\bsub[- ]?limits?\b", re.I),
        re.compile(r"\broom rent\b", re.I),
        re.compile(r"\bnot exceeding\b", re.I),
        re.compile(r"\bcapped at\b", re.I),
    ),
    "claim_procedure": (
        re.compile(r"\bclaim form\b", re.I),
        re.compile(r"\bclaim documents?\b", re.I),
        re.compile(r"\bcashless\b", re.I),
        re.compile(r"\breimbursement\b", re.I),
        re.compile(r"\bintimation\b", re.I),
    ),
    "coverage": (
        re.compile(r"\bcovered\b", re.I),
        re.compile(r"\bcoverage\b", re.I),
        re.compile(r"\bbenefits?\b", re.I),
        re.compile(r"\bwe will (pay|cover|reimburse|indemnify)\b", re.I),
        re.compile(r"\bhospitali[sz]ation expenses\b", re.I),
    ),
    "definition": (
        re.compile(r"\bshall mean\b", re.I),
        re.compile(r"\bis defined as\b", re.I),
        re.compile(r"\brefers to\b", re.I),
    ),
}

LIST_ITEM_START = re.compile(
    r"^\s*(?:\(?[ivxlc]+[).]|\(?[a-z][).]|\d+[).])\s+",
    re.I,
)


def normalize_text(text: str) -> str:
    """Normalize common PDF characters and whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return text.replace("\xa0", " ")


def is_boilerplate(line: str) -> bool:
    """Return True for obvious repeated headers and contact details."""
    lowered = line.lower()
    if any(pattern.lower() in lowered for pattern in settings.chunk_skip_patterns):
        return True
    return any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS)


def is_section_start(line: str) -> bool:
    """Return True when a line looks like a section heading."""
    stripped = line.strip()
    return bool(stripped) and any(
        pattern.match(stripped) for pattern in SECTION_PATTERNS
    )


def _clean_lines(text: str) -> list[str]:
    """Return useful non-empty lines from extracted PDF text."""
    lines = []
    for line in normalize_text(text).splitlines():
        stripped = " ".join(line.split())
        if not stripped or is_boilerplate(stripped):
            continue
        lines.append(stripped)
    return lines


def split_into_sections(text: str) -> list[dict]:
    """Group page text under simple section headings when available."""
    lines = _clean_lines(text)
    if not lines:
        return []

    sections = []
    heading = ""
    body = []

    for line in lines:
        if is_section_start(line):
            if body:
                sections.append({"section": heading, "text": " ".join(body)})
            heading = line[:100]
            body = [line]
        else:
            body.append(line)

    if body:
        sections.append({"section": heading, "text": " ".join(body)})

    return sections


def _hard_split(text: str, max_size: int) -> list[str]:
    """Split an oversized sentence by words as a safe fallback."""
    pieces = []
    current = []
    current_length = 0

    for word in text.split():
        added = len(word) + (1 if current else 0)
        if current and current_length + added > max_size:
            pieces.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length += added

    if current:
        pieces.append(" ".join(current))
    return pieces


def split_long_text(text: str, max_size: int) -> list[str]:
    """Split text at sentence boundaries without exceeding the target size."""
    if len(text) <= max_size:
        return [text]

    pieces = []
    current = ""

    for sentence in SENTENCE_BOUNDARY.split(text):
        if len(sentence) > max_size:
            if current:
                pieces.append(current)
                current = ""
            pieces.extend(_hard_split(sentence, max_size))
            continue

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
    """Merge very small chunks so retrieval receives useful context."""
    if not pieces:
        return []

    merged = []
    for piece in pieces:
        if merged and len(piece["text"]) < min_size:
            merged[-1]["text"] = (
                f"{merged[-1]['text']} {piece['text']}".strip()
            )
            continue
        merged.append(dict(piece))

    if len(merged) > 1 and len(merged[0]["text"]) < min_size:
        first = merged.pop(0)
        merged[0]["text"] = f"{first['text']} {merged[0]['text']}".strip()

    return merged


def detect_topics(text: str) -> list[str]:
    """Return matching retrieval topics, or general when none match."""
    topics = [
        topic
        for topic, patterns in TOPIC_PATTERNS.items()
        if any(pattern.search(text) for pattern in patterns)
    ]
    return topics or ["general"]


def _inherit_list_topics(
    records: list[dict],
    carried_topics: list[str] | None,
) -> None:
    """Give unlabelled list items the previous strong topic."""
    active = carried_topics

    for record in records:
        topics = record["metadata"]["topics"]
        if topics != ["general"]:
            active = topics
        elif active and LIST_ITEM_START.match(record["text"]):
            record["metadata"]["topics"] = list(active)
            record["metadata"]["topic"] = active[0]


def chunk_page(
    text: str,
    uin: str,
    insurer: str,
    product: str,
    page: int,
    carried_topics: list[str] | None = None,
) -> list[dict]:
    """Chunk one policy page and attach stable retrieval metadata."""
    pieces = []

    for section in split_into_sections(text):
        for chunk_text in split_long_text(
            section["text"],
            settings.max_chunk_size,
        ):
            pieces.append(
                {
                    "section": section["section"],
                    "text": chunk_text,
                }
            )

    records = []
    for piece in merge_short_pieces(pieces, settings.min_chunk_size):
        topic_text = f"{piece['section']} {piece['text']}"
        topics = (
            detect_topics(topic_text)
            if settings.tag_topics
            else ["general"]
        )

        records.append(
            {
                "text": piece["text"],
                "metadata": {
                    "uin": uin or "",
                    "insurer": insurer or "",
                    "product": product or "",
                    "page": page,
                    "section": piece["section"],
                    "topic": topics[0],
                    "topics": topics,
                    "chunk_type": "text",
                },
            }
        )

    _inherit_list_topics(records, carried_topics)
    logger.debug("page_chunked page=%s chunks=%s", page, len(records))
    return records


def last_topics(records: list[dict]) -> list[str] | None:
    """Return the last non-general topic for cross-page list continuity."""
    for record in reversed(records):
        topics = record.get("metadata", {}).get("topics") or []
        if topics and topics != ["general"]:
            return topics
    return None
