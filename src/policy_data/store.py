"""SQLite store for structured policy rules."""
from __future__ import annotations

from src.utils.sqlite_store import SqliteStore

from .models import SCHEMA


GENERIC_TREATMENT_WORDS = {
    "surgery",
    "surgeries",
    "surgical",
    "treatment",
    "treatments",
    "procedure",
    "procedures",
    "therapy",
    "expenses",
    "medical",
    "hospitalization",
    "hospitalisation",
    "care",
    "inpatient",
    "outpatient",
    "patient",
    "admission",
    "hospital",
    "other",
    "major",
}


def _identifying_words(name: str) -> set[str]:
    """Return meaningful words from a treatment name."""
    cleaned = (name or "").lower().replace("_", " ").replace("-", " ")
    return {
        word
        for word in cleaned.split()
        if len(word) > 3 and word not in GENERIC_TREATMENT_WORDS
    }


class PolicyDataStore(SqliteStore):
    """Read and write structured policy facts."""

    SCHEMA = SCHEMA
    LABEL = "policy_data"

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
        """Return the best matching sub-limit for a treatment."""
        query_words = _identifying_words(treatment)
        if not query_words:
            return None

        best_row = None
        best_overlap = 0

        for row in self.get_sub_limits(policy_uin):
            stored_words = _identifying_words(row["treatment"])
            if not stored_words:
                continue
            if not (stored_words <= query_words or query_words <= stored_words):
                continue

            overlap = len(stored_words & query_words)
            if overlap > best_overlap:
                best_row = row
                best_overlap = overlap

        return best_row

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
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policy_deductibles WHERE policy_uin = ?",
                (policy_uin,),
            ).fetchone()
        return dict(row) if row else None

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
                "waiting_period_type, page) VALUES (?, ?, ?, ?, ?, ?)",
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
                "age_max, page) VALUES (?, ?, ?, ?, ?, ?, ?)",
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

    def get_copayments(
        self,
        policy_uin: str,
        age: int | None = None,
    ) -> list[dict]:
        """Return co-pay rows, optionally filtered by age."""
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
