"""SQLite schema for the CRM layer: customers, policies and claims."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id     TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    email           TEXT,
    phone           TEXT,
    date_of_birth   TEXT,
    address         TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id       TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    policy_number   TEXT NOT NULL,
    policy_type     TEXT NOT NULL,
    policy_name     TEXT NOT NULL,
    sum_insured     REAL NOT NULL,
    premium         REAL,
    start_date      TEXT NOT NULL,
    end_date        TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'expired', 'lapsed')),
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id        TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL REFERENCES customers(customer_id),
    policy_id       TEXT NOT NULL REFERENCES policies(policy_id),
    claim_type      TEXT NOT NULL,
    claim_amount    REAL NOT NULL,
    claim_date      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'under_review')),
    -- What the insurer actually agreed to pay, after sub-limits and
    -- co-payment. Stays NULL until the claim is assessed.
    eligible_amount REAL,
    rejection_reason TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_policies_customer ON policies(customer_id);
CREATE INDEX IF NOT EXISTS idx_claims_customer ON claims(customer_id);
CREATE INDEX IF NOT EXISTS idx_claims_policy ON claims(policy_id);
"""
