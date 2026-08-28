"""SQLite schema for the human-in-the-loop audit trail.

WHY this table exists: the copilot recommends, the employee decides. If a
claims decision is ever questioned later, the trail has to show three things
separately -- what the system recommended, what the human actually did, and
whether the two agreed. Storing only the final outcome would lose exactly the
information an audit needs.

It lives in the CRM database so claim_id can be a real foreign key to claims.
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS claim_decisions (
    decision_id         TEXT PRIMARY KEY,
    claim_id            TEXT NOT NULL,
    customer_id         TEXT NOT NULL,
    policy_id           TEXT,

    -- What the AI proposed, kept verbatim and never overwritten by the
    -- employee's answer. ai_recommendation is the whole engine result as
    -- JSON (decision, amounts, reasoning, evidence with citations).
    ai_decision         TEXT NOT NULL
                        CHECK (ai_decision IN ('approve', 'reject', 'needs_more_info')),
    ai_payable_amount   REAL,
    ai_recommendation   TEXT NOT NULL,

    -- What the employee actually did. 'edit' means they accepted the claim
    -- but changed the amount or the reasoning.
    employee_decision   TEXT NOT NULL
                        CHECK (employee_decision IN ('approve', 'edit', 'reject')),
    employee_payable_amount REAL,
    employee_edits      TEXT,
    override_reason     TEXT,

    -- Free text the employee wrote on the claim form before analysing it
    -- (e.g. "patient asked to expedite"). Carried through so it is not lost
    -- once the review screen closes.
    notes               TEXT,

    -- agreed = did the human land on the same call as the AI? Stored rather
    -- than derived so a later schema change cannot silently rewrite history.
    agreed              INTEGER NOT NULL DEFAULT 0,

    decided_by          TEXT NOT NULL,
    decided_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_decisions_claim ON claim_decisions(claim_id);
CREATE INDEX IF NOT EXISTS idx_decisions_by ON claim_decisions(decided_by);
"""
