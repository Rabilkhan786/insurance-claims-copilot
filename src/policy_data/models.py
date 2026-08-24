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

CREATE TABLE IF NOT EXISTS policy_room_rent (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_uin          TEXT NOT NULL,
    insurer             TEXT,
    sum_insured_min     REAL,
    sum_insured_max     REAL,
    room_rent_limit     REAL,
    icu_limit           REAL,
    page                INTEGER,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS network_hospitals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    hospital_name       TEXT NOT NULL,
    city                TEXT,
    insurer             TEXT,
    cashless_status     TEXT NOT NULL DEFAULT 'available'
                        CHECK (cashless_status IN ('available', 'suspended', 'unavailable')),
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS day_care_procedures (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    procedure_name      TEXT NOT NULL,
    procedure_category  TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now'))
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
CREATE INDEX IF NOT EXISTS idx_room_rent_uin ON policy_room_rent(policy_uin);
CREATE INDEX IF NOT EXISTS idx_network_hospitals_name
    ON network_hospitals(hospital_name);
CREATE INDEX IF NOT EXISTS idx_day_care_name
    ON day_care_procedures(procedure_name);
"""
