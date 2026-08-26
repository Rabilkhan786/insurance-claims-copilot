"""The typed recommendation an employee reviews.

WHY the numbers are assembled here rather than asked of the model: the
deterministic engine has already decided the status and computed the payable
amount. Handing those to the LLM and asking for them back is a round trip
that can only lose -- it once returned a payable of Rs 41,753 where the
engine had calculated Rs 38,000, and an employee has no way to tell which
figure to trust.

So the split is strict, and it is enforced by construction rather than by
asking the prompt nicely:

    engine  ->  status, payable_amount, evidence, missing_information
    model   ->  reasoning (prose only)

The model cannot contradict a figure it is never asked to produce.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from src.eligibility import ELIGIBLE, INELIGIBLE, NEEDS_MORE_INFO


class ClaimExplanation(BaseModel):
    """What the model is asked to produce, via create_agent's response_format.

    WHY only one field: response_format is LangChain's structured-output
    contract, so the model's reply is validated JSON instead of free text --
    that is genuinely useful, it replaces manually scanning the message list
    for the last AIMessage. But the shape stops at reasoning. status and
    payable_amount are not asked of the model at all, on the same principle
    as ClaimRecommendation.from_engine(): a field the model is never asked to
    produce is a field it cannot get wrong. Giving it a status field to fill
    in, only to discard whatever it wrote, would just be a more elaborate way
    of asking it to guess.
    """

    reasoning: str = Field(
        description=(
            "Structure: 1) Recommend: approve/reject/needs more info, and "
            "why, in one sentence. 2) The payable amount and which "
            "deduction reduced it, or what is missing if there is none yet. "
            "3) The policy clauses that support it, each with its citation "
            "[Source: {insurer}, UIN: {uin}, Page {page}]. 4) Anything "
            "missing, or 'None'."
        )
    )

# The engine says what is true of the claim; the employee-facing recommendation
# says what to do about it. They are different vocabularies on purpose.
ENGINE_STATUS_TO_RECOMMENDATION = {
    ELIGIBLE: "approve",
    INELIGIBLE: "reject",
    NEEDS_MORE_INFO: "needs_more_info",
}


class Evidence(BaseModel):
    """One policy clause, with everything a citation needs."""

    text: str
    uin: str | None = None
    insurer: str | None = None
    page: int | None = None

    @property
    def is_citable(self) -> bool:
        """A clause with no UIN or page must not be stated as a fact."""
        return bool(self.uin and self.page)

    def citation(self) -> str:
        """Render the one citation format used everywhere in this project."""
        return f"[Source: {self.insurer}, UIN: {self.uin}, Page {self.page}]"


class PolicyFactView(BaseModel):
    """One fact the engine resolved, and whether it was actually established."""

    name: str
    status: Literal["found", "not_applicable", "unknown"]
    value: float | int | bool | None = None
    source: str
    detail: str = ""


class ClaimRecommendation(BaseModel):
    """What the copilot proposes, for a human to accept, edit or overturn.

    Never a decision -- `status` is a recommendation, and nothing downstream
    acts on it without an employee's answer recorded alongside it.
    """

    status: Literal["approve", "reject", "needs_more_info"]
    payable_amount: float | None = Field(
        default=None,
        description="From the engine. None when no amount could be computed.",
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
        """Build the recommendation from an engine result plus the model's prose."""
        return cls(
            status=ENGINE_STATUS_TO_RECOMMENDATION.get(
                eligibility.get("status"), "needs_more_info"
            ),
            payable_amount=eligibility.get("estimated_payable"),
            bill_amount=eligibility.get("bill_amount") or 0,
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
