# AI Health Insurance Claims Copilot

An internal copilot for claims employees assessing health insurance claims.

The employee enters claim details. The system loads the customer's policy and claim history from SQLite, retrieves supporting clauses from policy PDFs with hybrid RAG, runs a deterministic eligibility and payable-amount check, and produces a recommendation with citations. The employee then approves, edits, or rejects the recommendation.

The AI does not make the final claim decision. The recommendation and the employee's decision are both stored for audit.

## Architecture

```text
Claim
  ↓
SQLite: customer, policy, claim history
  ↓
Hybrid RAG: policy clauses + citations
  ↓
Deterministic eligibility engine
  ↓
LangChain agent: explanation and tool use
  ↓
LangGraph human review
  ↓
Audit: AI recommendation + employee decision
```

### Core design

| Component | Responsibility |
|---|---|
| SQLite | Customer, policy, claim history, and curated policy facts |
| Pinecone hybrid RAG | Policy wording, coverage, exclusions, definitions, and citations |
| Eligibility engine | Eligibility checks and payable-amount calculation |
| LangChain | Agent and tool calling |
| LangGraph | Stateful workflow, checkpointing, and human review |
| FastAPI | JSON API |
| Streamlit | Employee interface |
| pytest | Deterministic claim-correctness tests |
| RAGAS | Retrieval evaluation |

The eligibility engine returns three states: `eligible`, `ineligible`, and `needs_more_info`. Policy facts are tracked as `found`, `not_applicable`, or `unknown`, so missing evidence does not silently become a default value.

## Claim calculation

For eligible claims, the deterministic engine applies policy rules in this order:

```text
sub-limit
  ↓
co-pay
  ↓
deductible
  ↓
remaining sum insured
```

Money and date calculations are handled by Python logic rather than the LLM.

## Tech stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Package manager | uv |
| Agent | LangChain `create_agent` |
| Orchestration | LangGraph `StateGraph`, `interrupt()` |
| LLM | Groq `openai/gpt-oss-120b` |
| Vector database | Pinecone |
| Embeddings | `BAAI/bge-base-en-v1.5` |
| Reranker | `BAAI/bge-reranker-base` |
| Database | SQLite |
| PDF parsing | PyMuPDF |
| API | FastAPI |
| UI | Streamlit |
| Evaluation | pytest + RAGAS |

## Setup

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Set the required environment variables:

```text
GROQ_API_KEY=
PINECONE_API_KEY=
```

Seed the demo database:

```bash
uv run python Data/seed.py
```

## Run the application

Streamlit employee UI:

```bash
uv run streamlit run streamlit_app.py
```

FastAPI:

```bash
uv run uvicorn app:app --port 8000
```

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health/readiness check |
| POST | `/review-claim` | Assess a claim and pause for employee review |
| POST | `/submit-decision` | Save the employee decision and resume the workflow |
| POST | `/chat` | Ask policy questions; supports streaming |
| POST | `/reset-memory` | Clear a conversation thread |

## Index policy PDFs

Add or rebuild the Pinecone index with:

```bash
uv run python main.py
uv run python main.py --reset
```

`--reset` clears the namespace before indexing. Use it after changing chunking or topic-tagging rules, or when documents have been removed from the corpus.

## Testing

Run the full test suite:

```bash
uv run pytest
```

Run the claim-correctness benchmark:

```bash
uv run pytest tests/test_claims_evaluation.py -v
```

The claim dataset contains 15 labelled cases covering sub-limits, co-pays, deductibles, waiting periods, exclusions, exhausted sum insured, lapsed policies, customer/policy mismatch, and `needs_more_info` cases.

## RAG evaluation

Install the optional evaluation dependencies:

```bash
uv sync --extra evaluation
```

Run the evaluation:

```bash
uv run python evaluation/run_ragas.py
```

Smoke run:

```bash
uv run python evaluation/run_ragas.py --smoke
```

The evaluation measures faithfulness, answer relevancy, context precision, and context recall. The baseline results are stored in `evaluation/baseline_results.json`.

## Project structure

```text
insurance-claims-copilot/
├── app.py
├── streamlit_app.py
├── main.py
├── config/
│   ├── config.yaml
│   └── settings.py
├── Data/
│   ├── insurance_documents/
│   ├── excluded_documents/
│   └── seed.py
├── src/
│   ├── agent/
│   │   ├── agent.py
│   │   ├── workflow.py
│   │   └── recommendation.py
│   ├── eligibility/
│   │   ├── engine.py
│   │   ├── facts.py
│   │   └── parsing.py
│   ├── tools/
│   │   ├── crm_tools.py
│   │   ├── rag_tools.py
│   │   └── calc_tools.py
│   ├── ingestion/
│   │   ├── page_parser.py
│   │   ├── table_classifier.py
│   │   ├── table_converter.py
│   │   ├── chunker.py
│   │   └── pipeline.py
│   ├── retrieval/
│   │   └── retrievers.py
│   ├── embeddings/
│   │   └── bge.py
│   ├── vectorstores/
│   │   └── pinecone_store.py
│   ├── cache/
│   │   └── rag_cache.py
│   ├── crm/
│   ├── policy_data/
│   ├── decisions/
│   └── utils/
├── evaluation/
│   ├── claims_dataset.json
│   ├── rag_dataset.json
│   ├── run_ragas.py
│   └── baseline_results.json
└── tests/
```

## Limitations

This repository is a portfolio/demo system rather than a production insurance platform.

- Demo customers and policies are seeded locally.
- The policy corpus does not cover every possible insurance document or clause.
- Some policy facts depend on the available curated data or retrieved wording; unresolved facts return `needs_more_info`.
- Some retrieval cases, especially long natural-language queries and exclusion clauses, still need improvement.
- The API is not configured with production authentication or authorization.
- The project does not provide legal, regulatory, or insurer-approved claim decisions.

## License

This repository is intended for demonstration and portfolio use.
