"""Run RAGAS evaluation for the insurance-policy retrieval dataset."""
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

from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_groq import ChatGroq  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

from config import settings  # noqa: E402
from src.embeddings import get_embedder  # noqa: E402
from src.policy_data import get_policy_store  # noqa: E402
from src.tools.rag_tools import (  # noqa: E402
    check_coverage,
    check_exclusion,
    check_waiting_period,
)
from src.utils import configure_logging  # noqa: E402

DATASET_PATH = ROOT / "evaluation" / "rag_dataset.json"
BASELINE_PATH = ROOT / "evaluation" / "baseline_results.json"

MAX_TOKENS = 2048
MAX_CONTEXT_CHARS = 12_000
ANSWER_DELAY_SECONDS = 10
EVAL_DELAY_SECONDS = 20
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 60

METRICS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "context_precision": context_precision,
    "context_recall": context_recall,
}

TOPICS = [
    "waiting_period",
    "coverage",
    "exclusion",
    "sub_limit",
    "copay_deductible",
]

PROMPT = """You are a health insurance assistant evaluating policy documents.
Answer only with facts explicitly stated in the context.
Do not add general knowledge or unsupported inferences.
If the context does not answer the question, reply exactly:
The context does not contain information about this.

CONTEXT:
{context}"""


def _patch_ragas_compat() -> None:
    import types
    import nest_asyncio

    nest_asyncio.apply = lambda *args, **kwargs: None
    stub = types.ModuleType("langchain_community.chat_models.vertexai")
    stub.ChatVertexAI = type("ChatVertexAI", (), {})
    sys.modules.setdefault("langchain_community.chat_models.vertexai", stub)


_patch_ragas_compat()


def _model():
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to evaluate")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=0,
        max_tokens=MAX_TOKENS,
    )


def _load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _pick(entries: list[dict], per_topic: int) -> list[dict]:
    return [
        entry
        for topic in TOPICS
        for entry in [e for e in entries if e.get("topic") == topic][:per_topic]
    ]


def _retrieve(entry: dict) -> list[dict]:
    question = entry["question"]
    topic = entry["topic"]
    uin = entry["context_uin"]

    if topic == "waiting_period":
        return check_waiting_period.invoke({"condition": question, "policy_uin": uin})
    if topic == "exclusion":
        return check_exclusion.invoke(
            {
                "treatment": f"{question} excluded not covered exclusion clause",
                "policy_uin": uin,
            }
        )
    return check_coverage.invoke({"treatment": question, "policy_uin": uin})


def _waiting_period_fact(entry: dict) -> str | None:
    question = entry["question"].lower()
    for row in get_policy_store().get_waiting_periods(entry["context_uin"]):
        condition = (row.get("condition") or "").lower()
        if condition and condition in question:
            return (
                f"Under policy UIN {entry['context_uin']} (page {row.get('page')}), the "
                f"waiting period for {row['condition']} is {row['waiting_period_months']} months."
            )
    return None


def _truncate(contexts: list[str]) -> list[str]:
    result: list[str] = []
    remaining = MAX_CONTEXT_CHARS
    for context in contexts:
        if remaining <= 0:
            break
        result.append(context[:remaining])
        remaining -= len(result[-1])
    return result


def _answer(model, question: str, contexts: list[str]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [("system", PROMPT), ("human", "{question}")]
    )
    return model.invoke(
        prompt.format_messages(context="\n\n".join(contexts), question=question)
    ).content


def _build_samples(entries: list[dict]) -> list[SingleTurnSample]:
    model = _model()
    samples = []

    for index, entry in enumerate(entries, start=1):
        print(f"  [{index}/{len(entries)}] {entry['topic']}: {entry['question'][:70]}")
        if index > 1:
            time.sleep(ANSWER_DELAY_SECONDS)

        contexts = [hit["text"] for hit in _retrieve(entry)]
        if entry["topic"] == "waiting_period":
            fact = _waiting_period_fact(entry)
            if fact:
                contexts.insert(0, fact)
        contexts = _truncate(contexts)

        samples.append(
            SingleTurnSample(
                user_input=entry["question"],
                retrieved_contexts=contexts,
                response=_answer(model, entry["question"], contexts),
                reference=entry["ground_truth"],
            )
        )

    return samples


def _mean(values: list[float]) -> float:
    valid = [value for value in values if not math.isnan(value)]
    return sum(valid) / len(valid) if valid else float("nan")


def _score(samples: list[SingleTurnSample]) -> dict[str, float]:
    judge = LangchainLLMWrapper(_model())
    embeddings = LangchainEmbeddingsWrapper(get_embedder())
    config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=MAX_RETRIES,
        max_wait=RETRY_WAIT_SECONDS,
    )
    values = {name: [] for name in METRICS}

    for index, sample in enumerate(samples, start=1):
        print(f"\nEvaluating question {index}/{len(samples)}...")
        if index > 1:
            time.sleep(EVAL_DELAY_SECONDS)
        try:
            result = evaluate(
                dataset=EvaluationDataset(samples=[sample]),
                metrics=list(METRICS.values()),
                llm=judge,
                embeddings=embeddings,
                run_config=config,
                show_progress=False,
            )
            scores = dict(result._repr_dict)
            print(
                "  "
                + "  ".join(f"{name}={scores.get(name, float('nan')):.2f}" for name in METRICS)
            )
        except Exception as error:
            print(f"  Scoring failed: {error}")
            scores = {}

        for name in METRICS:
            values[name].append(scores.get(name, float("nan")))

    return {name: _mean(scores) for name, scores in values.items()}


def _save(scores: dict[str, float], size: int) -> None:
    BASELINE_PATH.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "dataset_size": size,
                "provider": "groq",
                "model": settings.llm_model,
                "note": "Answer and judge use the same model, so scores may show self-grading bias.",
                "scores": scores,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run one question per topic")
    args = parser.parse_args()

    configure_logging()
    entries = _pick(_load_dataset(), 1 if args.smoke else 2)

    print(f"Model: {settings.llm_model} (groq)")
    print(f"Running {len(entries)} questions")
    print("\nGenerating answers...")
    samples = _build_samples(entries)

    print("\nScoring with RAGAS...")
    scores = _score(samples)
    _save(scores, len(entries))

    print(f"\nSaved scores -> {BASELINE_PATH}")
    print(" | ".join(f"{name}: {scores[name]:.2f}" for name in METRICS))


if __name__ == "__main__":
    main()
