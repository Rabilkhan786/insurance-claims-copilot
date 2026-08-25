# Evaluation

Run RAGAS evaluation over the 20-question dataset:

```bash
uv run python evaluation/run_ragas.py
```

Requires the `evaluation` optional dependencies:

```bash
uv sync --extra evaluation
```

Results are written to `evaluation/final_results.json`. The baseline is in
`evaluation/baseline_results.json`.
