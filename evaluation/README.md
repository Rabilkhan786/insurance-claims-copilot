# Evaluation

The thing worth measuring here is whether the copilot assesses a claim
correctly -- the status it lands on, and the exact payable amount.

```bash
uv run pytest tests/test_claims_evaluation.py -v
```

`claims_dataset.json` holds 15 labelled claims covering sub-limits, co-pays,
deductibles (including a bill that sits exactly at the deductible), waiting
periods, pre-existing disease, an exclusion, an exhausted sum insured, a
lapsed policy, a treatment date outside the policy term, a customer/policy
mismatch, and both kinds of `needs_more_info` -- a claim that names no
procedure, and a policy whose wording neither covers nor excludes the
treatment. Each case declares the expected status (`eligible` /
`ineligible` / `needs_more_info`) and the expected payable amount to the
rupee.

**No LLM judge is involved.** A payable amount is either Rs 38,000 or it is
wrong, and asking a model whether Rs 41,753 is close enough would be a worse
test than `==`. Retrieval is fuzzy and needs a judge; arithmetic is not and
does not.

Ground truth comes from `Data/seed.py`, so a change there has to be reflected
here. The tests also assert two invariants across every case: a
`needs_more_info` result must name what is missing, and no fact marked
`unknown` may ever reach the arithmetic.
