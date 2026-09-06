"""Typed claim recommendations assembled from deterministic engine results."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.eligibility import ELIGIBLE, INELIGIBLE, NEEDS_MORE_INFO


class ClaimExplanation(BaseModel):
    """Structured prose returned by the claim-explanation agent."""

    reasoning: str = Field(
        description=(
            "Structure: 1) Recommend: approve/reject/needs more info, and "
            "why, in one sentence. 2) The payable amount and which deduction "
            "reduced it, or what is missing if there is none yet. 3) The "
            "policy clauses that support it, each with its citation "
            "[Source: {insurer}, UIN: {uin}, Page {page}]. 4) Anything "
            "missing, or 'None'."
        )
    )


ENGINE_STATUS_TO_RECOMMENDATION = {
    ELIGIBLE: "approve",
    INELIGIBLE: "reject",
    NEEDS_MORE_INFO: "needs_more_info",
}


class Evidence(BaseModel):
    """One policy clause, with everything needed for a citation."""

    text: str
    uin: str | None = None
    insurer: str | None = None
    page: int | None = None

    @property
    def is_citable(self) -> bool:
        return bool(self.uin and self.page)

    def citation(self) -> str:
        return f"[Source: {self.insurer}, UIN: {self.uin}, Page {self.page}]"


class PolicyFactView(BaseModel):
    """One fact resolved by the eligibility engine."""

    name: str
    status: Literal["found", "not_applicable", "unknown"]
    value: float | int | bool | None = None
    source: str
    detail: str = ""


class ClaimRecommendation(BaseModel):
    """A recommendation for human review, not a final claim decision."""

    status: Literal["approve", "reject", "needs_more_info"]
    payable_amount: float | None = Field(
        default=None,
        description="From the deterministic engine; None when no amount can be computed.",
    )
    bill_amount: float = 0
    reasoning: str = ""
    reason_summary: str = Field(
        default="",
        description="The engine's one-line account of what decided the claim.",
    )
    evidence: list[Evidence] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    facts: list[PolicyFactView] = Field(default_factory=list)
    deductions: dict = Field(default_factory=dict)

    @classmethod
    def from_engine(cls, eligibility: dict, reasoning: str) -> "ClaimRecommendation":
        """Build the employee-facing recommendation from engine output."""
        bill_amount = eligibility.get("bill_amount")
        return cls(
            status=ENGINE_STATUS_TO_RECOMMENDATION.get(
                eligibility.get("status"), "needs_more_info"
            ),
            payable_amount=eligibility.get("estimated_payable"),
            bill_amount=0 if bill_amount is None else bill_amount,
            reasoning=reasoning,
            reason_summary=eligibility.get("reason") or "",
            evidence=[
                Evidence(**{k: hit.get(k) for k in ("text", "uin", "insurer", "page")})
                for hit in eligibility.get("evidence") or []
                if hit.get("text")
            ],
            missing_information=eligibility.get("missing_information") or [],
            facts=[
                PolicyFactView(**fact)
                for fact in (eligibility.get("facts") or {}).values()
            ],
            deductions=eligibility.get("deductions") or {},
        )
