"""Run RAGAS evaluation for the insurance-policy retrieval dataset.

Usage:
    uv run python evaluation/run_ragas.py
    uv run python evaluation/run_ragas.py --smoke

The evaluator follows the same topic-specific retrieval tools used by the
claims agent, generates answers with the production Groq model, and scores
faithfulness, relevance, context precision, and context recall.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "evaluation" / "rag_dataset.json"
BASELINE_PATH = ROOT / "evaluation" / "baseline_results.json"

EVAL_MAX_TOKENS = 2048
MAX_CONTEXT_CHARS = 12_000
ANSWER_DELAY_SECONDS = 10
EVAL_DELAY_SECONDS = 20
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 60

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "context_precision": "Precision",
    "context_recall": "Recall",
}

EVAL_TOPICS = [
    "waiting_period",
    "coverage",
    "exclusion",
    "sub_limit",
    "copay_deductible",
]

EVAL_SYSTEM_PROMPT = """You are a health insurance assistant evaluating policy documents.
Answer only with facts explicitly stated in the context.
Do not add general knowledge or unsupported inferences.
If the context does not answer the question, reply exactly:
The context does not contain information about this.

CONTEXT:
{context}"""


def _groq_llm():
    from langchain_groq import ChatGroq
    from config import settings

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to evaluate")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=EVAL_MAX_TOKENS,
    )


def _patch_ragas_compat() -> None:
    """Keep RAGAS 0.2.15 importable in this project's environment."""
    import types
    import nest_asyncio

    nest_asyncio.apply = lambda *args, **kwargs: None
    stub = types.ModuleType("langchain_community.chat_models.vertexai")
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules.setdefault("langchain_community.chat_models.vertexai", stub)


_patch_ragas_compat()

from src.embeddings import get_embedder  # noqa: E402
from src.policy_data import get_policy_store  # noqa: E402
from src.tools.rag_tools import (  # noqa: E402
    check_coverage,
    check_exclusion,
    check_waiting_period,
)
from src.utils import configure_logging  # noqa: E402


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def pick_balanced(entries: list[dict], n_per_topic: int) -> list[dict]:
    return [
        entry
        for topic in EVAL_TOPICS
        for entry in [e for e in entries if e.get("topic") == topic][:n_per_topic]
    ]


def _retrieve(question: str, topic: str, policy_uin: str) -> list[dict]:
    if topic == "waiting_period":
        return check_waiting_period.invoke(
            {"condition": question, "policy_uin": policy_uin}
        )
    if topic == "exclusion":
        return check_exclusion.invoke(
            {
                "treatment": f"{question} excluded not covered exclusion clause",
                "policy_uin": policy_uin,
            }
        )
    return check_coverage.invoke({"treatment": question, "policy_uin": policy_uin})


def _sql_waiting_period_fact(policy_uin: str, question: str) -> str | None:
    for row in get_policy_store().get_waiting_periods(policy_uin):
        condition = (row.get("condition") or "").lower()
        if condition and condition in question.lower():
            return (
                f"Under policy UIN {policy_uin} (page {row.get('page')}), the "
                f"waiting period for {row['condition']} is "
                f"{row['waiting_period_months']} months."
            )
    return None


def _truncate_contexts(contexts: list[str]) -> list[str]:
    result: list[str] = []
    total = 0
    for chunk in contexts:
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        result.append(chunk[:remaining])
        total += min(len(chunk), remaining)
        if len(chunk) > remaining:
            break
    return result


def _generate_answer(model, question: str, context: str) -> str:
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", EVAL_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    return model.invoke(prompt.format_messages(context=context, question=question)).content


def build_samples(entries: list[dict]) -> list:
    from ragas import SingleTurnSample

    model = _groq_llm()
    samples = []
    for index, entry in enumerate(entries, start=1):
        print(f"  [{index}/{len(entries)}] {entry['topic']}: {entry['question'][:70]}")
        if index > 1:
            time.sleep(ANSWER_DELAY_SECONDS)

        hits = _retrieve(entry["question"], entry["topic"], entry["context_uin"])
        contexts = [hit["text"] for hit in hits]
        if entry["topic"] == "waiting_period":
            fact = _sql_waiting_period_fact(entry["context_uin"], entry["question"])
            if fact:
                contexts.insert(0, fact)

        contexts = _truncate_contexts(contexts)
        answer = _generate_answer(model, entry["question"], "\n\n".join(contexts))
        samples.append(
            SingleTurnSample(
                user_input=entry["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=entry["ground_truth"],
            )
        )
    return samples


def _score_one(sample, judge_llm, judge_embeddings, metrics, run_config) -> dict:
    from ragas import EvaluationDataset, evaluate

    result = evaluate(
        dataset=EvaluationDataset(samples=[sample]),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
        show_progress=False,
    )
    return dict(result._repr_dict)


def run_metrics(samples: list) -> dict[str, float]:
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    judge_llm = __import__(
        "ragas.llms", fromlist=["LangchainLLMWrapper"]
    ).LangchainLLMWrapper(_groq_llm())
    judge_embeddings = LangchainEmbeddingsWrapper(get_embedder())
    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=MAX_RETRIES,
        max_wait=RETRY_WAIT_SECONDS,
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    values: dict[str, list[float]] = {metric: [] for metric in METRIC_LABELS}

    for index, sample in enumerate(samples, start=1):
        print(f"\nEvaluating question {index}/{len(samples)}...")
        if index > 1:
            time.sleep(EVAL_DELAY_SECONDS)
        try:
            scores = _score_one(sample, judge_llm, judge_embeddings, metrics, run_config)
            print(
                "  "
                + "  ".join(
                    f"{label}={scores.get(metric, float('nan')):.2f}"
                    for metric, label in METRIC_LABELS.items()
                )
            )
        except Exception as error:
            print(f"  Scoring failed: {error}")
            scores = {}

        for metric in METRIC_LABELS:
            values[metric].append(scores.get(metric, float("nan")))

    def mean(scores: list[float]) -> float:
        valid = [score for score in scores if not math.isnan(score)]
        return sum(valid) / len(valid) if valid else float("nan")

    return {metric: mean(scores) for metric, scores in values.items()}


def save_results(scores: dict[str, float], dataset_size: int) -> None:
    from config import settings

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_size": dataset_size,
        "provider": "groq",
        "model": settings.llm_model,
        "note": "Answer and judge use the same model, so scores may show self-grading bias.",
        "scores": scores,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run one question per topic")
    args = parser.parse_args()

    configure_logging()
    per_topic = 1 if args.smoke else 2
    entries = pick_balanced(load_dataset(), per_topic)

    print(f"Model: {settings.llm_model} (groq)")
    print(f"Running {len(entries)} questions")
    print("\nGenerating answers...")
    samples = build_samples(entries)

    print("\nScoring with RAGAS...")
    scores = run_metrics(samples)
    save_results(scores, len(entries))

    print(f"\nSaved scores -> {BASELINE_PATH}")
    print(
        " | ".join(
            f"{label}: {scores.get(metric, float('nan')):.2f}"
            for metric, label in METRIC_LABELS.items()
        )
    )


if __name__ == "__main__":
    from config import settings
    main()
