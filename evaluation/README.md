# Evaluation

Two things are measured separately, because they fail in different ways and
only one of them is a judgement call.

| What | Dataset | Runner | Scored by |
|---|---|---|---|
| **Claim correctness** | `claims_dataset.json` (15 claims) | `tests/test_claims_evaluation.py` | Exact comparison |
| **Retrieval quality** | `rag_dataset.json` (10 questions) | `run_ragas.py` | RAGAS, LLM judge |

## Claim correctness — the main benchmark

This is the thing the product does: take a claim, decide it, and put a number
on it. Every case declares the expected status (`eligible` / `ineligible` /
`needs_more_info`) and the expected payable amount to the rupee.

```bash
uv run pytest tests/test_claims_evaluation.py -v
```

**No LLM judge is involved.** A payable amount is either Rs 38,000 or it is
wrong, and asking a model whether Rs 41,753 is close enough would be a worse
test than `==`.

The 15 cases cover: sub-limits, co-pays, deductibles (including a bill that
sits exactly at the deductible), waiting periods, pre-existing disease, an
exclusion, an exhausted sum insured, a lapsed policy, a treatment date outside
the policy term, a customer/policy mismatch, and both kinds of
`needs_more_info` — a claim that names no procedure, and a policy whose wording
neither covers nor excludes the treatment.

Ground truth comes from `Data/seed.py`, so a change there has to be reflected
here. The tests also assert two invariants across every case: a
`needs_more_info` result must name what is missing, and no fact marked
`unknown` may ever reach the arithmetic.

## Retrieval quality — RAGAS

```bash
uv sync --extra evaluation
uv run python evaluation/run_ragas.py           # all 10
uv run python evaluation/run_ragas.py --smoke   # 1 per topic
```

Scores are written to `baseline_results.json`. The committed file holds a
5-question smoke run against the current code — the previous baseline was
deleted rather than kept, because it had been measured before the retrieval
fix and no longer described anything in this repository.

`rag_dataset.json` holds two questions for each kind of clause the eligibility
engine actually reads: waiting periods, coverage, exclusions, sub-limits, and
co-pay/deductible wording. Each question was written from the real PDF text
with the source UIN and page recorded.

Retrieval is routed by topic through the same tools the live agent uses — a
waiting-period question goes to `check_waiting_period`, an exclusion question
to `check_exclusion` — rather than through one generic search the product
never runs.

| Metric | Question it answers |
|---|---|
| Faithfulness | Is every statement in the answer supported by the retrieved text? |
| Answer relevancy | Does the answer address the question actually asked? |
| Context precision | Of the chunks retrieved, how many were relevant? |
| Context recall | Did retrieval find everything the reference answer needs? |

### Reading the scores honestly

**Faithfulness rewards abstention.** A refusal asserts nothing, so it scores
1.00. A run where retrieval fails often can therefore post a *higher*
faithfulness than one that answers more questions. Read it next to recall.

**The judge marks its own homework.** Answers and judging both run
`gpt-oss-120b` on Cerebras (on two separate keys, so the rate-limit buckets
are independent). Self-grading bias inflates faithfulness and relevancy.
Treat the numbers as directional, not absolute.

**`nan` is a missing score, not a zero.** A judge call that hits a rate limit
is recorded as `nan` and excluded from the mean, so a metric may be averaged
over fewer questions than were run. The per-question lines printed during the
run show which.

## Why RAGAS does not score the claims

RAGAS grades text against retrieved context. It has no way to check that
Rs 40,000 capped by a 5% co-pay leaves Rs 38,000, and no business ruling on
it. Retrieval is fuzzy and needs a judge; arithmetic is not and does not.
