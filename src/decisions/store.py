"""SQLite-backed audit trail: what the AI recommended vs what the employee did."""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from uuid import uuid4

from .models import SCHEMA

logger = logging.getLogger(__name__)

# The three calls the employee can make on a recommendation.
EMPLOYEE_DECISIONS = ("approve", "edit", "reject")


def _ai_decision_from(recommendation: dict) -> str:
    """Turn the eligibility engine's result into the AI's headline call.

    The engine answers eligible True/False. "needs_more_info" is the third
    case: it ran, but could not find the evidence to justify either answer.
    """
    if recommendation.get("needs_more_info"):
        return "needs_more_info"
    return "approve" if recommendation.get("eligible") else "reject"


class DecisionStore:
    """Thin CRUD layer over the claim_decisions audit table."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        logger.info("decisions_schema_ready path=%s", self.db_path)

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
                    recommendation.get("estimated_payable"),
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
