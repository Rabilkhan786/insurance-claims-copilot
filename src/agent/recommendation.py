"""Models for claim recommendations shown to the employee."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.eligibility import ELIGIBLE, INELIGIBLE, NEEDS_MORE_INFO


class ClaimExplanation(BaseModel):
    """Structured explanation generated from the engine result."""

    reasoning: str = Field(
        description=(
            "Explain the recommendation, payable amount or missing data, "
            "supporting policy citations, and any follow-up needed."
        )
    )


ENGINE_STATUS_TO_RECOMMENDATION = {
    ELIGIBLE: "approve",
    INELIGIBLE: "reject",
    NEEDS_MORE_INFO: "needs_more_info",
}


class Evidence(BaseModel):
    """One retrieved policy clause used as evidence."""

    text: str
    uin: str | None = None
    insurer: str | None = None
    page: int | None = None


class PolicyFactView(BaseModel):
    """One policy fact resolved by the eligibility engine."""

    name: str
    status: Literal["found", "not_applicable", "unknown"]
    value: float | int | bool | None = None
    source: str
    detail: str = ""


class ClaimRecommendation(BaseModel):
    """AI recommendation that must still be reviewed by an employee."""

    status: Literal["approve", "reject", "needs_more_info"]
    payable_amount: float | None = None
    bill_amount: float = 0
    reasoning: str = ""
    reason_summary: str = ""
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
                Evidence(
                    **{
                        key: hit.get(key)
                        for key in ("text", "uin", "insurer", "page")
                    }
                )
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
