"""SQLite schema for the claim decision audit trail."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS claim_decisions (
    decision_id         TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    customer_id         TEXT NOT NULL,
    policy_id           TEXT,

    ai_decision         TEXT NOT NULL
                        CHECK (ai_decision IN ('approve', 'reject', 'needs_more_info')),
    ai_payable_amount   REAL,
    ai_recommendation   TEXT NOT NULL,

    employee_decision   TEXT NOT NULL
                        CHECK (employee_decision IN ('approve', 'edit', 'reject')),
    employee_payable_amount REAL,
    employee_edits      TEXT,
    override_reason     TEXT,
    notes               TEXT,
    agreed              INTEGER NOT NULL DEFAULT 0,
    decided_by          TEXT NOT NULL,
    decided_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_claim
    ON claim_decisions(claim_id);
CREATE INDEX IF NOT EXISTS idx_decisions_by
    ON claim_decisions(decided_by);
"""
