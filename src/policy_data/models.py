"""SQLite schema for structured facts pulled out of the policy PDFs.

WHY these are SQL and not just Pinecone: the eligibility engine has to do
exact lookups and arithmetic on them. "What is the cataract sub-limit under
this UIN" must be a SELECT, not a similarity search -- the LLM never does
the calculation.

Rows are populated from the tables the ingestion pipeline classified as
sql_and_pinecone or sql_only (see artifacts/staged_tables.json).
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS policy_sub_limits (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin      TEXT NOT NULL,
    insurer         TEXT,
    treatment       TEXT NOT NULL,
    limit_amount    REAL,
    -- 'amount' for a flat rupee cap, 'percent_si' for a share of the
    -- sum insured -- they are calculated very differently.
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
    -- Many co-pays only bite above a certain age, so the band matters.
    age_min         INTEGER,
    age_max         INTEGER,
    page            INTEGER,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_sub_limits_uin ON policy_sub_limits(policy_uin);
CREATE INDEX IF NOT EXISTS idx_waiting_periods_uin
    ON policy_waiting_periods(policy_uin);
CREATE TABLE IF NOT EXISTS policy_deductibles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin        TEXT NOT NULL,
    insurer           TEXT,
    -- Top-up and super top-up plans pay only above this threshold. The value
    -- is chosen on the policy schedule, not fixed by the wording, so it is
    -- stored per UIN rather than read out of the document text.
    deductible_amount REAL NOT NULL,
    page              INTEGER,
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_copayments_uin ON policy_copayments(policy_uin);
CREATE INDEX IF NOT EXISTS idx_deductibles_uin ON policy_deductibles(policy_uin);
"""
