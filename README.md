# AI Health Insurance Claims Copilot

A portfolio project that helps a health-insurance claims employee review a claim using customer records, policy documents, deterministic calculations, and an AI explanation.

The AI does **not** make the final claim decision. It produces a recommendation, and the employee can approve, edit, or reject it.

## Problem

A claims employee may need to check several things before deciding a health-insurance claim:

- Is the policy active?
- Does the policy belong to this customer?
- Is the treatment covered?
- Is the treatment excluded?
- Is a waiting period still active?
- Is there a treatment sub-limit?
- Is there a co-payment or deductible?
- How much sum insured is still available?

The information is split between customer records and long policy PDFs. This project brings those checks into one workflow.

## Project flow

```text
Employee enters a claim
        ↓
SQLite CRM
Customer + policy + previous claims
        ↓
Hybrid RAG
Relevant policy clauses + citations
        ↓
Deterministic eligibility engine
Eligibility + payable amount
        ↓
LLM explanation
Clear recommendation with evidence
        ↓
LangGraph human review
Approve / Edit / Reject
        ↓
SQLite audit trail
AI recommendation + employee decision
```

## How a claim is reviewed

1. The employee selects a customer and policy and enters the claim details.
2. The application loads customer, policy, and previous-claim data from SQLite.
3. Relevant policy clauses are retrieved from Pinecone using dense and sparse search.
4. The deterministic eligibility engine checks policy validity, coverage, exclusions, waiting periods, sub-limits, co-payments, deductibles, and remaining sum insured.
5. Python calculates the payable amount. The LLM does not calculate insurance money values.
6. The LLM explains the engine result using the retrieved policy evidence.
7. LangGraph pauses the workflow so the employee can approve, edit, or reject the recommendation.
8. The AI recommendation and employee decision are stored for audit.

## Claim calculation

For an eligible claim, deductions are applied in this order:

```text
Claim amount
    ↓
Sub-limit
    ↓
Co-payment
    ↓
Deductible
    ↓
Remaining sum insured
    ↓
Payable amount
```

Keeping this calculation in normal Python makes the result predictable and testable.

## Eligibility results

The engine returns one of three states:

- `eligible` — required checks passed and a payable amount can be calculated.
- `ineligible` — a definite rule blocks the claim, such as an exclusion or expired policy.
- `needs_more_info` — an important policy fact could not be established safely.

Policy facts are also tracked as `found`, `not_applicable`, or `unknown`. This prevents missing information from silently becoming a default value.

## Main components

| Component | Purpose |
|---|---|
| SQLite | Stores demo customers, policies, previous claims, policy facts, and reviewed decisions |
| Pinecone | Stores policy-document chunks for retrieval |
| BGE embeddings | Dense semantic retrieval |
| Pinecone sparse index | Keyword/lexical retrieval |
| Cross-encoder reranker | Reranks dense + sparse retrieval results |
| Eligibility engine | Runs deterministic insurance checks and calculations |
| LangChain tools | Give the agent controlled access to CRM, RAG, and calculation functions |
| LangGraph | Manages claim-review state and human approval |
| FastAPI | JSON API |
| Streamlit | Employee-facing demo interface |
| pytest | Tests deterministic logic and workflow behaviour |
| RAGAS | Optional RAG evaluation |

## Project structure

```text
insurance-claims-copilot/
├── app.py                  # FastAPI API
├── streamlit_app.py        # Employee UI
├── main.py                 # Policy PDF indexing entry point
├── config/
│   ├── config.yaml         # Non-secret configuration
│   └── settings.py         # Loads YAML + environment variables
├── Data/
│   ├── insurance_documents/ # Source policy PDFs
│   └── seed.py              # Creates demo SQLite data
├── src/
│   ├── agent/              # LLM agent, recommendation model, LangGraph workflow
│   ├── crm/                # Customer, policy, and claim database access
│   ├── decisions/          # Human-review audit trail
│   ├── eligibility/        # Deterministic claim checks
│   ├── tools/              # CRM, RAG, and calculation tools
│   ├── ingestion/          # PDF parsing, chunking, and table handling
│   ├── retrieval/          # Hybrid retrieval and reranking
│   ├── embeddings/         # BGE embedding model
│   ├── vectorstores/       # Pinecone dense/sparse storage
│   ├── policy_data/        # Structured policy-rule storage
│   ├── cache/              # Retrieval cache
│   └── utils/              # Shared SQLite and logging helpers
├── evaluation/             # Claim and RAG evaluation datasets
└── tests/                  # Automated tests
```

`Data/` contains source/demo assets committed to the project. Lowercase `data/` is created at runtime for SQLite database files and is ignored by Git.

## Demo data

`Data/seed.py` creates a small synthetic dataset for predictable demonstrations and tests. The demo cases cover:

- a normal covered claim,
- an incomplete waiting period,
- a nearly exhausted sum insured,
- an excluded treatment,
- an expired policy,
- and a policy whose wording is not available to RAG.

Run:

```bash
uv run python Data/seed.py
```

## Setup

Requirements: Python 3.11 and `uv`.

```bash
uv sync
cp .env.example .env
```

Add the required keys to `.env`:

```text
GROQ_API_KEY=
PINECONE_API_KEY=
```

Seed the demo database:

```bash
uv run python Data/seed.py
```

Index the policy PDFs:

```bash
uv run python main.py
```

Use `--reset` when the indexed document set or chunking rules have changed:

```bash
uv run python main.py --reset
```

## Run the application

Streamlit UI:

```bash
uv run streamlit run streamlit_app.py
```

FastAPI:

```bash
uv run uvicorn app:app --port 8000
```

Main API endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check API and retrieval readiness |
| POST | `/review-claim` | Analyse a claim and pause for employee review |
| POST | `/submit-decision` | Save the employee decision and resume the workflow |
| POST | `/chat` | Ask customer-scoped policy and claim questions |
| POST | `/reset-memory` | Clear one chat session |

## Testing

Run the test suite:

```bash
uv run pytest
```

The claim evaluation dataset includes cases for policy validity, waiting periods, exclusions, sub-limits, co-payments, deductibles, remaining sum insured, ownership checks, and missing information.

## RAG evaluation

RAGAS is optional and kept separate from the application dependencies.

```bash
uv sync --extra evaluation
uv run python evaluation/run_ragas.py --smoke
```

The evaluation tracks faithfulness, answer relevancy, context precision, and context recall.

## Important design decisions

**Why not let the LLM calculate claim amounts?**  
Insurance calculations should be deterministic. Python applies the policy rules; the LLM only explains the result.

**Why use SQLite and RAG together?**  
SQLite is used for exact structured facts such as customer records, previous claims, and curated policy values. RAG is used for policy wording that needs semantic retrieval and citations.

**Why keep a human review step?**  
The copilot is decision support. The employee remains responsible for the final claim decision, and both the recommendation and final decision are saved.

## Limitations

This is a portfolio/demo system, not a production insurance platform.

- Customer and claim records are synthetic demo data.
- The policy corpus does not represent every insurer or policy clause.
- Some claims correctly return `needs_more_info` when evidence is unavailable.
- Retrieval quality can still vary for long or unusual policy wording.
- Production authentication, authorization, monitoring, and regulatory controls are not implemented.
- The project does not provide legal or insurer-approved claim decisions.
