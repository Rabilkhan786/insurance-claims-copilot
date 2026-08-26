# AI Health Insurance Claims Copilot

An internal tool for **claims employees** assessing health insurance claims.
It is not a customer-facing chatbot — customers never touch this system.

An employee enters a claim (customer, amount, treatment, diagnosis, hospital,
date). The system pulls the customer's policy from SQL, retrieves the
supporting clauses from real IRDAI policy PDFs, runs a deterministic
eligibility checklist, and shows a **recommendation**: a status, a payable
amount, the reasoning, and the cited clauses behind it. The employee then
approves, edits or rejects it.

**The AI never finalises a claim.** Every recommendation is stored alongside
what the employee actually decided, so an audit can see both and tell whether
the human agreed.

```
claim ─▶ SQL: customer, policy, claim history
      ─▶ RAG: policy clauses, with citations
      ─▶ deterministic engine: eligibility + payable amount
      ─▶ LLM: explains the engine's result
      ─▶ HUMAN: approve / edit / reject          ◀── the decision happens here
      ─▶ audit trail: recommendation AND decision, side by side
```

## Why each piece is there

| Piece | What it is responsible for |
|---|---|
| **SQL** (SQLite) | Customers, policies, claim history, and curated policy facts (sub-limits, waiting periods, co-pays, deductibles). Exact lookups that must never be approximate. |
| **RAG** (Pinecone hybrid) | Unstructured policy wording — coverage, exclusions, definitions — retrieved with the insurer, UIN and page needed to cite it. |
| **Deterministic engine** | Decides eligibility and computes the payable amount. Pure Python. No LLM involved in any money or date arithmetic. |
| **LangChain** (`create_agent`) | Tool calling: the agent chooses which lookups to run and writes the explanation an employee reads. |
| **LangGraph** (`StateGraph`) | The stateful claim workflow, and the `interrupt()` that pauses it for human review and resumes it on the same thread. |
| **The human** | The decision. Always. |
| **Evaluation** | RAGAS for retrieval quality; pytest for claim correctness; LangSmith for agent and workflow traces. |

The split between the engine and the model is the design decision the whole
project turns on. The model is good at reading a clause and explaining a
result; it is unreliable at arithmetic and at deciding. So it never does
either — it is handed a finished decision and asked to make it readable.

## Three answers, not two

The engine answers `eligible`, `ineligible`, or **`needs_more_info`**.

That third state matters. "No exclusion clause matched" is not evidence that a
treatment is covered, and an empty co-pay lookup is not evidence that the
co-pay is zero. Every policy fact the engine resolves carries its own status:

| Fact status | Meaning |
|---|---|
| `found` | We have the value, and where it came from (records or wording). |
| `not_applicable` | We read the policy and it states no such condition. |
| `unknown` | We could not establish it, so we cannot say either way. |

Any `unknown` that could change the outcome sends the whole claim to
`needs_more_info` with a named list of what to chase — instead of quietly
defaulting to zero and showing the employee a confident figure built on a fact
nobody ever established.

## What it does

- Assesses a claim against the customer's actual policy, citing insurer, UIN
  and page for every policy-derived fact
- Applies sub-limit → co-pay → deductible → remaining sum insured, in that
  order, deterministically
- Computes the exact date a waiting period ends
- Pauses for human review, then records the recommendation and the decision
- Says "I could not find this in the policy documents" rather than answering
  from general knowledge

## Tech stack

| Layer | Choice |
|---|---|
| Language / packaging | Python 3.11, [uv](https://docs.astral.sh/uv/) |
| Agent | LangChain `create_agent` 1.3 (9 tools) |
| Orchestration | LangGraph 1.2 (`StateGraph`, subgraph-as-node, `interrupt()`) |
| Checkpointing | `SqliteSaver` (`data/checkpoints.db`) |
| LLM | Groq `openai/gpt-oss-120b` |
| Vector DB | Pinecone — dense + hosted sparse, hybrid |
| Embeddings | `BAAI/bge-base-en-v1.5` via `langchain-huggingface` |
| Reranker | `BAAI/bge-reranker-base` cross-encoder |
| Structured data | SQLite (`data/crm.db`, including the `claim_decisions` audit trail) |
| PDF parsing | PyMuPDF |
| API | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS (retrieval) + pytest (claim correctness) |

## Setup

Needs Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

Copy the environment template and fill in your keys:

```bash
cp .env.example .env
```

`.env` needs:

```
GROQ_API_KEY=
PINECONE_API_KEY=
LANGSMITH_API_KEY=
CEREBRAS_API_KEY=
CEREBRAS_JUDGE_API_KEY=
```

`CEREBRAS_*` are only needed to run the RAGAS evaluation. `LANGSMITH_API_KEY`
is optional; tracing switches itself off without it.

Create and seed the demo databases:

```bash
uv run python Data/seed.py
```

## Running it

The Streamlit app is the product:

```bash
uv run streamlit run streamlit_app.py
```

Pick a customer and policy, enter the claim, and review what comes back.

The JSON API, if you want it instead:

```bash
uv run uvicorn app:app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `POST /review-claim` | Assess a claim; returns a recommendation and pauses for review |
| `POST /submit-decision` | Resume the paused thread with the employee's decision |
| `POST /chat` | Free-form policy question (set `"stream": true` for token streaming) |
| `POST /reset-memory` | Forget one conversation thread |
| `GET /health` | Readiness |

## Indexing the PDFs

```bash
uv run python main.py           # add to the existing index
uv run python main.py --reset   # wipe the namespace first, then rebuild
```

Document IDs hash the chunk's own content, so a plain run overwrites the
vectors it produces rather than duplicating them. It cannot know about chunks
produced from a PDF you have since deleted — use `--reset` for that, and after
any change to how chunks are tagged, because tags are baked in at index time.

## Evaluation

Two separate things are measured, because they fail in different ways.

### Claim correctness — pytest

`evaluation/claims_dataset.json` holds 15 labelled claims covering sub-limits,
co-pays, deductibles, waiting periods, exclusions, an exhausted sum insured, a
lapsed policy, a customer/policy mismatch, and both kinds of
`needs_more_info`. Each case declares the expected status and payable amount.

```bash
uv run pytest tests/test_claims_evaluation.py -v
```

A payable amount is either right or wrong, so it is checked with `==`. No LLM
judge is involved in scoring money or dates.

### Retrieval quality — RAGAS

`evaluation/rag_dataset.json` holds 10 policy-evidence questions, two each for
waiting periods, coverage, exclusions, sub-limits, and co-pay/deductible
wording. Every question was written from the actual PDF text with its UIN and
page recorded.

```bash
uv run sync --extra evaluation
uv run python evaluation/run_ragas.py           # full run
uv run python evaluation/run_ragas.py --smoke   # 1 per topic
```

Scores are written to `evaluation/baseline_results.json`. A full run takes
roughly 20 minutes: it spaces calls out to stay inside the provider's rate
limit.

RAGAS measures faithfulness, answer relevancy, context precision and context
recall — retrieval, not decisions.

The committed `baseline_results.json` is a 5-question smoke run (one per
topic) against the current code: faithfulness 1.00, relevancy 0.64, precision
0.45, recall 0.60. Read those next to the caveats in
[evaluation/README.md](evaluation/README.md) — faithfulness of 1.00 partly
reflects that an abstention asserts nothing, and one of the five questions
abstained. The previous baseline was deleted rather than kept: it predated the
retrieval fix below and no longer described this code.

### Agent traces — LangSmith

Set `LANGSMITH_API_KEY` and tool selection, tool arguments, and the LangGraph
workflow trace are all visible per run. Useful for spotting a wasted or
repeated tool call.

## Known limitations

Stated plainly, because a portfolio project that hides them is worse than one
that does not.

- **Only the seeded demo customers exist.** Five customers and five policies,
  in `Data/seed.py`. The policy *facts* seeded for them (sub-limits, waiting
  periods, co-pay) are transcribed from the real PDFs, with one exception,
  marked in the file: the New India top-up deductible is a representative
  demo figure, because that wording does not fix a single amount.
- **Not every policy has curated rows.** A claim on a policy with no rows
  falls back to reading the number out of the wording. When the wording does
  not state it either, the claim becomes `needs_more_info` rather than being
  assessed on a default.
- **Corpus gaps.** Restore/refill and organ-donor clauses appear in fewer than
  half the PDFs. Coverage, exclusions, waiting periods, sub-limits and co-pay
  are present in 83–100% of them.
- **20 of the 21 PDFs are indexed.** `25.SmartHealth Group Insurance` prints no
  UIN anywhere in its text, and a citation needs one. Ingestion skips it rather
  than inventing a UIN-shaped string from the filename, which is what it used
  to do — the model then cited that string as though it were real.
- **Exclusion retrieval is the weakest path.** The clauses are indexed and
  correctly tagged, but the cross-encoder sometimes ranks them below more
  general clauses from the same document. This needs a better reranker or a
  lexical boost, not a config change.
- **Long questions retrieve worse than short ones.** Measured, not guessed:
  querying `check_coverage` with `"robotic surgery"` puts the right clause at
  rank 1, while passing the whole RAGAS question sentence pushes it out of the
  top 8 entirely — the extra words dilute the embedding and the reranker
  prefers generic prose. This mostly affects `/chat`, where a user's sentence
  becomes the query. The claim path is unaffected: it queries with the
  treatment field, which is already short. **The RAGAS harness has not been
  tuned around this** — it still passes the full question, so the published
  scores include the weakness rather than hiding it.
- **This is not a licensed insurance product.** Results are estimates from
  demo records and public documents, and carry no weight with any insurer.

### Recently fixed

Table-derived chunks were indexed under the *table's* type (`modern_treatment`,
`copayment`, `room_rent`) while the retrieval tools filter on the *clause*
topics the chunker uses (`coverage`, `sub_limit`, `copay`). Those two
vocabularies never met, so entire tables sat in Pinecone — correct, citable,
and unreachable by any tool. A "Robotic surgeries covered up to Rs X" row
could not be found by a robotic-surgery coverage query. `TABLE_TYPE_TOPICS` in
`src/ingestion/table_classifier.py` now maps one vocabulary onto the other.

Separately, the Pinecone upsert retry helper had been deleted in an earlier
refactor while its two call sites remained, so indexing raised `NameError` on
the first batch. It is now a `tenacity` policy.

## Project structure

```
health-agentic-rag/
  app.py                      FastAPI JSON API
  streamlit_app.py            Employee UI (no business logic)
  main.py                     Index the PDFs into Pinecone
  config/
    config.yaml               Every tunable value
    settings.py               Typed, validated settings loader
  Data/
    insurance_documents/      21 IRDAI policy PDFs (20 indexed - see below)
    excluded_documents/       Removed from the corpus, with reasons
    seed.py                   Builds and seeds the demo databases
  src/
    agent/
      agent.py                create_agent, 9 tools, Context, prompt
      workflow.py             StateGraph: eligibility -> agent -> review -> persist
      recommendation.py       ClaimRecommendation, the typed output contract
    eligibility/
      engine.py               The deterministic checklist
      facts.py                PolicyFact: found / not_applicable / unknown
      parsing.py              Reads rupee amounts and month counts from clauses
    tools/                    The 9 @tool functions
      crm_tools.py            Customer, policies, claims
      rag_tools.py            Coverage, exclusions, waiting periods
      calc_tools.py           Deterministic money and date arithmetic
    ingestion/
      page_parser.py          Splits prose from tables by bounding box
      table_classifier.py     Routes each table; maps table type -> topics
      table_converter.py      Turns table rows into citable sentences
      chunker.py              Section-aware, multi-label topic tagging
      pipeline.py             Orchestrates a full index run
    retrieval/retrievers.py   EnsembleRetriever + CrossEncoderReranker
    embeddings/bge.py         HuggingFaceEmbeddings
    vectorstores/             Pinecone dense + hosted sparse
    cache/rag_cache.py        TTL cache, keyed by query + topic + UIN
    crm/, policy_data/,
    decisions/                SQLite models and stores
    utils/sqlite_store.py     Shared connection and schema bootstrap
  evaluation/
    claims_dataset.json       15 labelled claims - the main benchmark
    rag_dataset.json          10 policy-evidence questions
    run_ragas.py              The RAGAS runner
  tests/
```

## How it fits together

**Two layers.** `src/agent/agent.py` is LangChain's `create_agent` — it owns
the tool-calling loop and nothing here hand-rolls it. `src/agent/workflow.py`
is a LangGraph `StateGraph` with the compiled agent added as a node. The
workflow owns persistence; the agent inherits its checkpointer as a subgraph.

**Human review is `interrupt()`.** No hand-rolled "awaiting approval" flag. The
graph stops, checkpoints itself, and hands back the recommendation. The UI
resumes the same `thread_id` with `Command(resume=...)`. Everything before the
interrupt is a pure function of state, because it runs twice.

**Customer scoping is structural.** `customer_id` arrives as runtime `Context`
and reaches tools through `ToolRuntime`, so it never appears in the schema the
model sees. The model cannot pass another customer's ID because it cannot pass
one at all.

**The recommendation is assembled, not asked for.** `ClaimRecommendation` takes
its status, payable amount, evidence and missing-information list from the
engine, and only the prose from the model. The model cannot contradict a figure
it is never asked to produce.

**Citations run end to end.** Every chunk is tagged with `insurer`, `uin`,
`page` and `topics` at ingestion; table rows are rewritten as sentences with
the citation baked into the text; retrieval filters on that metadata *before*
searching; the engine carries it into its evidence; and the prompt enforces
`[Source: {insurer}, UIN: {uin}, Page {page}]` at generation time.

## Disclaimer

A demo and portfolio project. Not a licensed insurance product, and it makes
no real insurance decisions. Eligibility results are estimates from seeded
demo records and public documents. The customer data is fabricated. The policy
PDFs are real public IRDAI documents — confirm anything important against your
own policy wording and your insurer.
