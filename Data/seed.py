"""Populate every SQLite table with realistic demo data.

Run it with:  uv run python data/seed.py

Four customers are hand-designed to cover the four scenarios the eligibility
engine (Prompt 3) has to handle correctly:

    CUST001  happy path                  -- claim well within cover, approved
    CUST002  waiting period not complete -- maternity claimed too early
    CUST003  sum insured nearly exhausted -- most of the cover already used
    CUST004  rejected claim              -- treatment is on the exclusion list

Their policies reference REAL UINs pulled from the 20 PDFs in
Data/insurance_documents/, and the policy_data facts below (sub-limits,
waiting periods, co-payments, deductibles) are transcribed from those same
PDFs rather than invented, so a citation like "UIN SHAHLIP22027V032122,
page 8" is honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running this file directly, not just as a module.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.decisions import DecisionStore  # noqa: E402
from src.crm import CRMStore  # noqa: E402
from src.policy_data import PolicyDataStore  # noqa: E402

# ---------------------------------------------------------------------------
# CRM: customers, policies and claims
# ---------------------------------------------------------------------------
CUSTOMERS = [
    {
        "customer_id": "CUST001",
        "name": "Priya Sharma",
        "email": "priya.sharma@example.com",
        "phone": "+91-98200-11111",
        "date_of_birth": "1994-03-15",
        "address": "204 Lotus Apartments, Andheri West, Mumbai, Maharashtra",
    },
    {
        "customer_id": "CUST002",
        "name": "Rohan Mehta",
        "email": "rohan.mehta@example.com",
        "phone": "+91-98200-22222",
        "date_of_birth": "1990-07-02",
        "address": "17 Green Park Colony, Sector 21, Gurgaon, Haryana",
    },
    {
        "customer_id": "CUST003",
        "name": "Lakshmi Iyer",
        "email": "lakshmi.iyer@example.com",
        "phone": "+91-98200-33333",
        "date_of_birth": "1961-02-10",
        "address": "9 Kaveri Nagar, R.S. Puram, Coimbatore, Tamil Nadu",
    },
    {
        "customer_id": "CUST004",
        "name": "Arjun Nair",
        "email": "arjun.nair@example.com",
        "phone": "+91-98200-44444",
        "date_of_birth": "1981-03-22",
        "address": "45 Marine Drive Society, Kochi, Kerala",
    },
]

POLICIES = [
    {
        # HAPPY PATH -- Star Health Arogya Sanjeevani, 24-hour hospitalisation
        # claim well inside cover.
        "policy_id": "POL001",
        "customer_id": "CUST001",
        "policy_number": "SHAHLIP22027V032122",
        "policy_type": "individual",
        "policy_name": "Star Health Arogya Sanjeevani Policy",
        "sum_insured": 500000,
        "premium": 8500,
        "start_date": "2025-09-01",
        "end_date": "2026-08-31",
        "status": "active",
    },
    {
        # WAITING PERIOD NOT COMPLETE -- Oriental Happy Family Floater,
        # started 3 months ago. Real maternity waiting period is 24 months
        # (see policy_data seed below), so this claim is far too early.
        "policy_id": "POL002",
        "customer_id": "CUST002",
        "policy_number": "IRDAII/HLT/OIC/P-H/V.II/450/15-16",
        "policy_type": "family_floater",
        "policy_name": "Oriental Happy Family Floater Policy-2015",
        "sum_insured": 500000,
        "premium": 14200,
        "start_date": "2026-05-16",
        "end_date": "2027-05-15",
        "status": "active",
    },
    {
        # SUM INSURED NEARLY EXHAUSTED -- Star Health Senior Citizens Red
        # Carpet, three approved claims already used most of the cover.
        "policy_id": "POL003",
        "customer_id": "CUST003",
        "policy_number": "SHAHLIP25027V072425",
        "policy_type": "senior",
        "policy_name": "Star Health Senior Citizens Red Carpet Policy",
        "sum_insured": 1000000,
        "premium": 31000,
        "start_date": "2025-09-01",
        "end_date": "2026-08-31",
        "status": "active",
    },
    {
        # REJECTED CLAIM -- New India Top-Up Mediclaim, claim for a
        # cosmetic procedure that is explicitly excluded.
        "policy_id": "POL004",
        "customer_id": "CUST004",
        "policy_number": "IRDA/NL-HLT/NIA/P-H/V.I/35/14-15",
        "policy_type": "top_up",
        "policy_name": "New India Top-Up Mediclaim Policy",
        "sum_insured": 500000,
        "premium": 6200,
        "start_date": "2025-11-01",
        "end_date": "2026-10-31",
        "status": "active",
    },
]

CLAIMS = [
    {
        "claim_id": "CLM001",
        "customer_id": "CUST001",
        "policy_id": "POL001",
        "claim_type": "hospitalization",
        "claim_amount": 45000,
        "claim_date": "2026-06-10",
        "status": "approved",
        "eligible_amount": 45000,
        "rejection_reason": None,
    },
    {
        "claim_id": "CLM002",
        "customer_id": "CUST002",
        "policy_id": "POL002",
        "claim_type": "maternity",
        "claim_amount": 60000,
        "claim_date": "2026-08-10",
        "status": "rejected",
        "eligible_amount": None,
        "rejection_reason": (
            "Maternity waiting period not complete. Policy started "
            "2026-05-16; the maternity waiting period is 24 months, so "
            "cover begins 2028-05-16."
        ),
    },
    # Three approved claims this policy year, adding up to Rs 9,20,000 of
    # the Rs 10,00,000 sum insured -- Rs 80,000 remains.
    {
        "claim_id": "CLM003",
        "customer_id": "CUST003",
        "policy_id": "POL003",
        "claim_type": "hospitalization",
        "claim_amount": 400000,
        "claim_date": "2025-10-05",
        "status": "approved",
        "eligible_amount": 400000,
        "rejection_reason": None,
    },
    {
        "claim_id": "CLM004",
        "customer_id": "CUST003",
        "policy_id": "POL003",
        "claim_type": "hospitalization",
        "claim_amount": 350000,
        "claim_date": "2026-01-20",
        "status": "approved",
        "eligible_amount": 350000,
        "rejection_reason": None,
    },
    {
        "claim_id": "CLM005",
        "customer_id": "CUST003",
        "policy_id": "POL003",
        "claim_type": "hospitalization",
        "claim_amount": 170000,
        "claim_date": "2026-05-12",
        "status": "approved",
        "eligible_amount": 170000,
        "rejection_reason": None,
    },
    {
        "claim_id": "CLM006",
        "customer_id": "CUST004",
        "policy_id": "POL004",
        "claim_type": "cosmetic_surgery",
        "claim_amount": 80000,
        "claim_date": "2026-07-22",
        "status": "rejected",
        "eligible_amount": None,
        "rejection_reason": (
            "Cosmetic surgery is excluded under this policy unless "
            "necessitated by an accident requiring hospitalisation."
        ),
    },
]

def seed_crm(store: CRMStore) -> dict[str, int]:
    """Insert customers, policies and claims."""
    for customer in CUSTOMERS:
        store.add_customer(**customer)
    for policy in POLICIES:
        store.add_policy(**policy)
    for claim in CLAIMS:
        store.add_claim(**claim)

    return {
        "customers": len(CUSTOMERS),
        "policies": len(POLICIES),
        "claims": len(CLAIMS),
    }


# ---------------------------------------------------------------------------
# Policy data: facts transcribed from the real PDFs for our 4 demo UINs.
# ---------------------------------------------------------------------------
SUB_LIMITS = [
    {
        "policy_uin": "SHAHLIP22027V032122",
        "insurer": "Star Health",
        "treatment": "Cataract",
        "limit_amount": 40000,
        "limit_type": "amount",
        "page": 8,
    },
    {
        "policy_uin": "SHAHLIP25027V072425",
        "insurer": "Star Health",
        "treatment": "Cataract",
        "limit_amount": 15000,
        "limit_type": "amount",
        "page": 3,
    },
    {
        "policy_uin": "SHAHLIP25027V072425",
        "insurer": "Star Health",
        "treatment": "All other major surgeries",
        "limit_amount": 60000,
        "limit_type": "amount",
        "page": 3,
    },
]

WAITING_PERIODS = [
    {
        "policy_uin": "SHAHLIP22027V032122",
        "insurer": "Star Health",
        "condition": "pre-existing disease",
        "waiting_period_months": 48,
        "waiting_period_type": "pre_existing",
        "page": 8,
    },
    {
        "policy_uin": "IRDAII/HLT/OIC/P-H/V.II/450/15-16",
        "insurer": "Oriental Insurance",
        "condition": "maternity",
        "waiting_period_months": 24,
        "waiting_period_type": "maternity",
        "page": 3,
    },
]

COPAYMENTS = [
    {
        "policy_uin": "SHAHLIP22027V032122",
        "insurer": "Star Health",
        "condition": "all claims",
        "copay_percent": 5,
        "age_min": None,
        "age_max": None,
        "page": 8,
    },
]

# Top-up plans pay only above a deductible. The New India Top-Up wording does
# not fix a single figure -- the amount is selected on the policy schedule --
# so this is a DEMO value representing one of the standard options, not a
# number extracted from the PDF.
DEDUCTIBLES = [
    {
        "policy_uin": "IRDA/NL-HLT/NIA/P-H/V.I/35/14-15",
        "insurer": "New India Assurance",
        "deductible_amount": 200000,
        "page": 12,
    },
]

def seed_policy_data(store: PolicyDataStore) -> dict[str, int]:
    """Insert sub-limits, waiting periods, co-pays, room rent, and lookups."""
    for row in SUB_LIMITS:
        store.add_sub_limit(**row)
    for row in WAITING_PERIODS:
        store.add_waiting_period(**row)
    for row in COPAYMENTS:
        store.add_copayment(**row)
    for row in DEDUCTIBLES:
        store.add_deductible(**row)


    return {
        "policy_sub_limits": len(SUB_LIMITS),
        "policy_waiting_periods": len(WAITING_PERIODS),
        "policy_copayments": len(COPAYMENTS),
    }


def main() -> None:
    """Wipe and rebuild every SQLite database with fresh demo data."""
    settings.crm_db_path.unlink(missing_ok=True)

    crm_counts = seed_crm(CRMStore(settings.crm_db_path))
    # DecisionStore shares the CRM database so claim_decisions.claim_id
    # sits alongside the claims it audits.
    DecisionStore(settings.crm_db_path)
    policy_counts = seed_policy_data(PolicyDataStore(settings.crm_db_path))

    all_counts = {**crm_counts, **policy_counts}
    table_count = len(all_counts)
    row_count = sum(all_counts.values())

    print(f"created {row_count} rows in {table_count} tables")
    for table, count in all_counts.items():
        print(f"  {table:<24} {count}")



if __name__ == "__main__":
    main()
