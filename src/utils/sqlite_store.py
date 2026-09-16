"""Shared SQLite connection and schema setup."""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class SqliteStore:
    """Base class used by the project's SQLite stores."""

    SCHEMA: str = ""
    LABEL: str = "store"

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection that returns dict-like rows."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _init_schema(self) -> None:
        """Create this store's tables when they do not exist."""
        with self._connect() as connection:
            connection.executescript(self.SCHEMA)
        logger.info("%s_schema_ready path=%s", self.LABEL, self.db_path)
