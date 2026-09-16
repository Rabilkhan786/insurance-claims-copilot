"""SQLite schema for structured policy facts."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_sub_limits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin      TEXT NOT NULL,
    insurer         TEXT,
    treatment       TEXT NOT NULL,
    limit_amount    REAL,
    limit_type      TEXT NOT NULL DEFAULT 'amount'
                    CHECK (limit_type IN ('amount', 'percent_si', 'per_day')),
    page            INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS policy_waiting_periods (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin              TEXT NOT NULL,
    insurer                 TEXT,
    condition               TEXT NOT NULL,
    waiting_period_months   INTEGER NOT NULL,
    waiting_period_type     TEXT NOT NULL DEFAULT 'specific'
                            CHECK (waiting_period_type IN
                                   ('initial', 'specific', 'pre_existing', 'maternity')),
    page                    INTEGER,
    created_at              TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS policy_copayments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin      TEXT NOT NULL,
    insurer         TEXT,
    condition       TEXT,
    copay_percent   REAL NOT NULL,
    age_min         INTEGER,
    age_max         INTEGER,
    page            INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS policy_deductibles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin        TEXT NOT NULL,
    insurer           TEXT,
    deductible_amount REAL NOT NULL,
    page              INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sub_limits_uin
    ON policy_sub_limits(policy_uin);
CREATE INDEX IF NOT EXISTS idx_waiting_periods_uin
    ON policy_waiting_periods(policy_uin);
CREATE INDEX IF NOT EXISTS idx_copayments_uin
    ON policy_copayments(policy_uin);
CREATE INDEX IF NOT EXISTS idx_deductibles_uin
    ON policy_deductibles(policy_uin);
"""
