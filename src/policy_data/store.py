"""SQLite-backed store for structured policy facts.

The eligibility engine reads everything here by policy UIN, so every
lookup takes a UIN and returns exact rows -- no similarity search.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .models import SCHEMA

logger = logging.getLogger(__name__)

# Words that appear in almost every treatment name, so they identify nothing
# on their own. Mirrors the list in src/eligibility/engine.py, which applies
# the same idea to clause text rather than to table rows.
GENERIC_TREATMENT_WORDS = {
    "surgery", "surgeries", "surgical", "treatment", "treatments",
    "procedure", "procedures", "therapy", "expenses", "medical",
    "hospitalization", "hospitalisation", "care", "inpatient",
    "outpatient", "patient", "admission", "hospital", "other", "major",
}


def _identifying_words(name: str) -> set[str]:
    """Return only the words in a treatment name that actually identify it."""
    cleaned = (name or "").lower().replace("_", " ").replace("-", " ")
    return {
        word
        for word in cleaned.split()
        if len(word) > 3 and word not in GENERIC_TREATMENT_WORDS
    }


class PolicyDataStore:
    """Thin CRUD layer over the policy fact tables."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)
        logger.info("policy_data_schema_ready path=%s", self.db_path)

    # -- sub-limits ------------------------------------------------------
    def add_sub_limit(
        self,
        policy_uin: str,
        insurer: str,
        treatment: str,
        limit_amount: float,
        page: int,
        limit_type: str = "amount",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO policy_sub_limits "
                "(policy_uin, insurer, treatment, limit_amount, limit_type, page) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (policy_uin, insurer, treatment, limit_amount, limit_type, page),
            )

    def get_sub_limits(self, policy_uin: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM policy_sub_limits WHERE policy_uin = ?",
                (policy_uin,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_sub_limit(self, policy_uin: str, treatment: str) -> dict | None:
        """Look up one treatment's sub-limit by name.

        WHY not a plain LIKE: wrapping the query in wildcards only matches when
        the STORED name is the longer string, so a bill reading "Cataract
        Surgery" silently missed the stored "Cataract" row and the 40,000 cap
        was never applied. Matching runs in both directions instead.

        Only identifying words count. Matching on a generic word such as
        "surgery" alone would make a cataract bill hit every surgical row.
        """
        query_words = _identifying_words(treatment)
        if not query_words:
            return None

        best_row = None
        best_overlap = 0
        for row in self.get_sub_limits(policy_uin):
            stored_words = _identifying_words(row["treatment"])
            if not stored_words:
                continue
            # Either name may be the more specific one.
            if not (stored_words <= query_words or query_words <= stored_words):
                continue
            overlap = len(stored_words & query_words)
            if overlap > best_overlap:
                best_row, best_overlap = row, overlap

        return best_row

    # -- deductibles ------------------------------------------------------
    def add_deductible(
        self,
        policy_uin: str,
        insurer: str,
        deductible_amount: float,
        page: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO policy_deductibles "
                "(policy_uin, insurer, deductible_amount, page) "
                "VALUES (?, ?, ?, ?)",
                (policy_uin, insurer, deductible_amount, page),
            )

    def find_deductible(self, policy_uin: str) -> dict | None:
        """Return the deductible for a top-up policy, or None if it has none.

        Only top-up and super top-up plans carry one; an ordinary indemnity
        policy has no row here and pays from the first rupee.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policy_deductibles WHERE policy_uin = ?",
                (policy_uin,),
            ).fetchone()
        return dict(row) if row else None

    # -- waiting periods --------------------------------------------------
    def add_waiting_period(
        self,
        policy_uin: str,
        insurer: str,
        condition: str,
        waiting_period_months: int,
        page: int,
        waiting_period_type: str = "specific",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO policy_waiting_periods "
                "(policy_uin, insurer, condition, waiting_period_months, "
                " waiting_period_type, page) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    policy_uin,
                    insurer,
                    condition,
                    waiting_period_months,
                    waiting_period_type,
                    page,
                ),
            )

    def get_waiting_periods(self, policy_uin: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM policy_waiting_periods WHERE policy_uin = ?",
                (policy_uin,),
            ).fetchall()
        return [dict(row) for row in rows]

    def find_waiting_period(self, policy_uin: str, condition: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policy_waiting_periods "
                "WHERE policy_uin = ? AND LOWER(condition) LIKE ?",
                (policy_uin, f"%{condition.lower()}%"),
            ).fetchone()
        return dict(row) if row else None

    # -- co-payments -------------------------------------------------------
    def add_copayment(
        self,
        policy_uin: str,
        insurer: str,
        condition: str,
        copay_percent: float,
        page: int,
        age_min: int | None = None,
        age_max: int | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO policy_copayments "
                "(policy_uin, insurer, condition, copay_percent, age_min, "
                " age_max, page) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    policy_uin,
                    insurer,
                    condition,
                    copay_percent,
                    age_min,
                    age_max,
                    page,
                ),
            )

    def get_copayments(self, policy_uin: str, age: int | None = None) -> list[dict]:
        """Return co-pay rules, narrowed to an age band when age is given."""
        query = "SELECT * FROM policy_copayments WHERE policy_uin = ?"
        params: list = [policy_uin]
        if age is not None:
            query += (
                " AND (age_min IS NULL OR age_min <= ?)"
                " AND (age_max IS NULL OR age_max >= ?)"
            )
            params.extend([age, age])

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]
