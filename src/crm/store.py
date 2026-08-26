"""SQLite-backed CRM store: customers, policies and claims."""
from __future__ import annotations

import logging

from src.utils.sqlite_store import SqliteStore

from .models import SCHEMA

logger = logging.getLogger(__name__)


class CRMStore(SqliteStore):
    """Thin CRUD layer over the CRM SQLite database."""

    SCHEMA = SCHEMA
    LABEL = "crm"

    # -- customers ---------------------------------------------------
    def add_customer(
        self,
        customer_id: str,
        name: str,
        email: str | None = None,
        phone: str | None = None,
        date_of_birth: str | None = None,
        address: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO customers "
                "(customer_id, name, email, phone, date_of_birth, address) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (customer_id, name, email, phone, date_of_birth, address),
            )

    def get_customer(self, customer_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM customers WHERE customer_id = ?",
                (customer_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_customers(self) -> list[dict]:
        """Every customer, for the claim form's picker.

        Here rather than in the UI because the UI was opening its own sqlite3
        connection and writing its own SELECT -- a second, silent copy of this
        table's shape that would not have moved if the schema did.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT customer_id, name FROM customers ORDER BY customer_id"
            ).fetchall()
        return [dict(row) for row in rows]

    # -- policies ------------------------------------------------------
    def add_policy(
        self,
        policy_id: str,
        customer_id: str,
        policy_number: str,
        policy_type: str,
        policy_name: str,
        sum_insured: float,
        start_date: str,
        end_date: str,
        premium: float | None = None,
        status: str = "active",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO policies "
                "(policy_id, customer_id, policy_number, policy_type, "
                " policy_name, sum_insured, premium, start_date, end_date, "
                " status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    policy_id,
                    customer_id,
                    policy_number,
                    policy_type,
                    policy_name,
                    sum_insured,
                    premium,
                    start_date,
                    end_date,
                    status,
                ),
            )

    def get_policies(self, customer_id: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM policies WHERE customer_id = ?",
                (customer_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_policy(self, policy_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policies WHERE policy_id = ?",
                (policy_id,),
            ).fetchone()
        return dict(row) if row else None

    # -- claims --------------------------------------------------------
    def add_claim(
        self,
        claim_id: str,
        customer_id: str,
        policy_id: str,
        claim_type: str,
        claim_amount: float,
        claim_date: str,
        status: str = "pending",
        eligible_amount: float | None = None,
        rejection_reason: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO claims "
                "(claim_id, customer_id, policy_id, claim_type, claim_amount, "
                " claim_date, status, eligible_amount, rejection_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    claim_id,
                    customer_id,
                    policy_id,
                    claim_type,
                    claim_amount,
                    claim_date,
                    status,
                    eligible_amount,
                    rejection_reason,
                ),
            )

    def get_claims(
        self,
        customer_id: str,
        policy_id: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM claims WHERE customer_id = ?"
        params: list[str] = [customer_id]
        if policy_id:
            query += " AND policy_id = ?"
            params.append(policy_id)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def get_claim_status(self, claim_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                # rejection_reason and eligible_amount are the two things a
                # customer actually asks about, so a status lookup must carry
                # them -- without them the agent can say "rejected" but never
                # why, or "approved" but not for how much.
                "SELECT claim_id, policy_id, claim_type, status, claim_amount, "
                "eligible_amount, claim_date, rejection_reason "
                "FROM claims WHERE claim_id = ?",
                (claim_id,),
            ).fetchone()
        return dict(row) if row else None

