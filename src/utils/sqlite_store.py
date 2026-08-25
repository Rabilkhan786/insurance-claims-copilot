"""Shared SQLite plumbing for the three stores in this project.

CRMStore, PolicyDataStore and DecisionStore each opened their own connection,
set the same row factory, and ran their own schema on first use -- the same
fifteen lines written three times. They differ only in which schema they
create and what they log, so those are the two things a subclass supplies.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SqliteStore:
    """Base class: one connection style, one schema bootstrap."""

    # Subclasses set these two.
    SCHEMA: str = ""
    LABEL: str = "store"

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection that returns rows as mappings, not tuples.

        foreign_keys is off by default in SQLite and has to be switched on per
        connection, otherwise a claim could reference a policy that does not
        exist.
        """
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        """Create the tables if they are not there yet."""
        with self._connect() as connection:
            connection.executescript(self.SCHEMA)
        logger.info("%s_schema_ready path=%s", self.LABEL, self.db_path)
