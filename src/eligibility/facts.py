"""How the engine represents a policy fact it may or may not actually know.

WHY this exists: the engine used to return plain numbers, so "this policy
states no co-payment" and "nobody could find out whether this policy states a
co-payment" arrived at the calculator as the same value -- None, read as 0%.
The claim was then priced as if the customer owed nothing, and the employee
was shown a confident figure built on a fact that was never established.

A PolicyFact keeps three things apart:

    FOUND           we have the value, and where it came from
    NOT_APPLICABLE  we searched the policy and it states no such condition
    UNKNOWN         we could not search, so we cannot say either way

The middle one is the distinction that matters. A curated waiting-period
schedule that does not list cataract is real evidence that cataract has no
waiting period. Retrieving nothing at all for a policy is not evidence of
anything -- and any UNKNOWN that could change the decision sends the whole
claim to needs_more_info instead of a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field

FOUND = "found"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"

# Where a value came from, for the employee reading the recommendation.
SOURCE_SQL = "policy records"
SOURCE_WORDING = "policy wording"
SOURCE_NONE = "not established"


@dataclass
class PolicyFact:
    """One policy fact, its provenance, and whether it is genuinely known."""

    name: str
    status: str
    value: float | int | None = None
    source: str = SOURCE_NONE
    detail: str = ""
    evidence: list[dict] = field(default_factory=list)

    @property
    def is_known(self) -> bool:
        """True when the fact was established, either as a value or as absent."""
        return self.status != UNKNOWN

    def value_or(self, default):
        """The value when found, otherwise the caller's default.

        Only safe to call once the fact is known -- an UNKNOWN fact must stop
        the claim, not quietly collapse to a default.
        """
        return self.value if self.status == FOUND else default

    def to_dict(self) -> dict:
        """Flatten for the API response, the UI, and the audit record."""
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "source": self.source,
            "detail": self.detail,
        }


def found(name: str, value, source: str, detail: str = "", evidence=None) -> PolicyFact:
    """A fact we have a value for."""
    return PolicyFact(name, FOUND, value, source, detail, list(evidence or []))


def not_applicable(name: str, detail: str, evidence=None) -> PolicyFact:
    """We looked at the policy and it states no such condition."""
    return PolicyFact(
        name, NOT_APPLICABLE, None, SOURCE_WORDING, detail, list(evidence or [])
    )


def unknown(name: str, detail: str) -> PolicyFact:
    """We could not establish this one way or the other."""
    return PolicyFact(name, UNKNOWN, None, SOURCE_NONE, detail)
