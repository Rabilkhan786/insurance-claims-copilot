"""Policy facts used by the deterministic eligibility engine."""
from __future__ import annotations

from dataclasses import dataclass, field

FOUND = "found"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"

SOURCE_SQL = "policy records"
SOURCE_WORDING = "policy wording"
SOURCE_NONE = "not established"


@dataclass
class PolicyFact:
    """Store a policy fact, its status, source, and supporting evidence."""

    name: str
    status: str
    value: float | int | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    evidence: list[dict] = field(default_factory=list)

    @property
    def is_known(self) -> bool:
        """Return True when the fact is established or not applicable."""
        return self.status != UNKNOWN

    def value_or(self, default):
        """Return the value when found, otherwise the supplied default."""
        return self.value if self.status == FOUND else default

    def to_dict(self) -> dict:
        """Convert the fact to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "source": self.source,
            "detail": self.detail,
        }


def found(
    name: str,
    value,
    source: str,
    detail: str = "",
    evidence=None,
) -> PolicyFact:
    """Create a fact with a known value."""
    return PolicyFact(name, FOUND, value, source, detail, list(evidence or []))


def not_applicable(name: str, detail: str, evidence=None) -> PolicyFact:
    """Create a fact that was checked and does not apply."""
    return PolicyFact(
        name,
        NOT_APPLICABLE,
        None,
        SOURCE_WORDING,
        detail,
        list(evidence or []),
    )


def unknown(name: str, detail: str) -> PolicyFact:
    """Create a fact that could not be established."""
    return PolicyFact(name, UNKNOWN, None, SOURCE_NONE, detail)
