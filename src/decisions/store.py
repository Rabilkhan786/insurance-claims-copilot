"""SQLite-backed audit trail: what the AI recommended vs what the employee did."""
from __future__ import annotations

import json
import logging
import sqlite3
from uuid import uuid4

from src.utils.sqlite_store import SqliteStore

from .models import SCHEMA

logger = logging.getLogger(__name__)

# The three calls the employee can make on a recommendation.
EMPLOYEE_DECISIONS = ("approve", "edit", "reject")

# What the copilot can recommend. Matches the CHECK constraint on the table
# and ClaimRecommendation.status -- one vocabulary, three places it is used.
AI_DECISIONS = ("approve", "reject", "needs_more_info")


def _ai_decision_from(recommendation: dict) -> str:
    """Read the AI's headline call off the recommendation it produced.

    ClaimRecommendation.status already speaks this vocabulary -- approve,
    reject, needs_more_info -- because the engine's three states were mapped
    into it when the recommendation was built. Nothing is re-derived here;
    re-deriving it was how the audit row and the screen once disagreed.
    """
    status = recommendation.get("status")
    return status if status in AI_DECISIONS else "needs_more_info"


class DecisionStore(SqliteStore):
    """Thin CRUD layer over the claim_decisions audit table."""

    SCHEMA = SCHEMA
    LABEL = "decisions"

    def record(
        self,
        claim_id: str,
        customer_id: str,
        recommendation: dict,
        employee_decision: str,
        decided_by: str,
        policy_id: str | None = None,
        employee_payable_amount: float | None = None,
        employee_edits: str | None = None,
        override_reason: str | None = None,
    ) -> str:
        """Save one reviewed recommendation and return its decision_id."""
        if employee_decision not in EMPLOYEE_DECISIONS:
            raise ValueError(f"employee_decision must be one of {EMPLOYEE_DECISIONS}")

        ai_decision = _ai_decision_from(recommendation)
        # "approve" is the only employee answer that means "as recommended".
        agreed = int(employee_decision == "approve" and ai_decision == "approve")
        decision_id = str(uuid4())

        self._insert(
            decision_id,
            claim_id,
            customer_id,
            policy_id,
            ai_decision,
            recommendation,
            employee_decision,
            employee_payable_amount,
            employee_edits,
            override_reason,
            agreed,
            decided_by,
        )
        print(
            f"decision recorded: claim={claim_id} ai={ai_decision} "
            f"employee={employee_decision} agreed={bool(agreed)}"
        )
        return decision_id

    def _insert(self, decision_id, claim_id, customer_id, policy_id, ai_decision,
                recommendation, employee_decision, employee_payable_amount,
                employee_edits, override_reason, agreed, decided_by) -> None:
        """Write the row. Split out so record() stays readable."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO claim_decisions ("
                "decision_id, claim_id, customer_id, policy_id, ai_decision, "
                "ai_payable_amount, ai_recommendation, employee_decision, "
                "employee_payable_amount, employee_edits, override_reason, "
                "agreed, decided_by) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    decision_id, claim_id, customer_id, policy_id, ai_decision,
                    recommendation.get("payable_amount"),
                    json.dumps(recommendation, default=str),
                    employee_decision, employee_payable_amount, employee_edits,
                    override_reason, agreed, decided_by,
                ),
            )

    def get(self, decision_id: str) -> dict | None:
        """Read one decision back, with the recommendation JSON parsed."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM claim_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        return self._to_dict(row)

    def recent(self, limit: int = 20) -> list[dict]:
        """The latest decisions across all claims, for the review queue."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM claim_decisions ORDER BY decided_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._to_dict(row) for row in rows]

    def agreement_rate(self) -> dict:
        """How often the employee accepted the AI's call, as-is.

        This is the number that tells you whether the copilot is actually
        helping or whether people are routinely overriding it.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total, SUM(agreed) AS agreed FROM claim_decisions"
            ).fetchone()

        total = row["total"] or 0
        agreed = row["agreed"] or 0
        rate = round(100 * agreed / total, 1) if total else 0.0
        return {"total": total, "agreed": agreed, "agreement_rate_percent": rate}

    def _to_dict(self, row: sqlite3.Row | None) -> dict | None:
        """Convert a row to a dict, parsing the stored recommendation JSON."""
        if row is None:
            return None
        record = dict(row)
        try:
            record["ai_recommendation"] = json.loads(record["ai_recommendation"])
        except (TypeError, ValueError):
            logger.warning("bad_recommendation_json decision_id=%s", record.get("decision_id"))
        return record
