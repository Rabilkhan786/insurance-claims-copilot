"""Run the RAGAS evaluation suite against evaluation/rag_dataset.json.

Usage:  uv run python evaluation/run_ragas.py
        uv run python evaluation/run_ragas.py --smoke   (1 per topic)

Scores retrieval quality (not claim correctness -- see
tests/test_claims_evaluation.py for that) with Groq's gpt-oss-120b acting as
both answerer and judge. Full explanation, caveats and self-grading-bias
tradeoff: evaluation/README.md. Results are saved to
evaluation/baseline_results.json.
"""
from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_fixed,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "evaluation" / "rag_dataset.json"
BASELINE_PATH = ROOT / "evaluation" / "baseline_results.json"

# gpt-oss-120b spends tokens on internal reasoning before it writes anything,
# so a small cap returns an empty string rather than a short answer.
EVAL_MAX_TOKENS = 2048


def _groq_llm(temperature: float = 0):
    """Build a chat model on the same provider, model and key the live app uses.

    Nothing evaluation-specific here on purpose: the point of scoring is to
    measure what actually ships, not a stand-in model on a different host.
    """
    from langchain_groq import ChatGroq

    from config import settings

    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is required to evaluate")
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=temperature,
        max_tokens=EVAL_MAX_TOKENS,
    )


# Capping retrieved context at 3,000 tokens keeps judge prompts reasonable.
MAX_CONTEXT_CHARS = 12_000

# Spacing between questions and between answers, to stay under Groq's
# per-minute limit -- one model doing both jobs means one limit to respect.
EVAL_DELAY_SECONDS = 20
ANSWER_DELAY_SECONDS = 10

# A 429 means the per-minute window is full; waiting it out clears it.
RETRY_WAIT_SECONDS = 60
MAX_RETRIES = 3


def _is_rate_limit(error: BaseException) -> bool:
    """True for a rate-limit refusal, whatever type it arrives wrapped in."""
    text = str(error).lower()
    return "429" in text or "rate_limit" in text


_retry_on_rate_limit = retry(
    retry=retry_if_exception(_is_rate_limit),
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_fixed(RETRY_WAIT_SECONDS),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
    reraise=True,
)

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "context_precision": "Precision",
    "context_recall": "Recall",
}

EVAL_TOPICS = ["waiting_period", "coverage", "exclusion", "sub_limit", "copay_deductible"]

# Stricter prompt for evaluation: every stated fact must come from the context.
# The production prompt allows conversational hedging that RAGAS faithfulness
# cannot verify against retrieved chunks, which deflates the score.
EVAL_SYSTEM_PROMPT = """You are a health insurance assistant evaluating policy documents.
Answer ONLY using facts explicitly stated in the CONTEXT below.
Do not add any information from general knowledge.
Do not hedge, guess, or infer beyond what the context says.
If the answer is not in the context, reply exactly: The context does not contain information about this.
Quote relevant numbers or policy terms exactly as they appear.

CONTEXT:
{context}"""


def _patch_ragas_compat() -> None:
    """Work around two RAGAS 0.2.15 environment incompatibilities."""
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


class _AnswerGenerator:
    """Groq gpt-oss-120b answer generator, strict eval-only prompt.

    Same model that judges the answers below -- see the module docstring for
    the self-grading-bias tradeoff that comes with that.
    """

    def __init__(self) -> None:
        self.model = _groq_llm()

    def answer(self, question: str, context: str) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        messages = prompt.format_messages(context=context, question=question)
        return _retry_on_rate_limit(self.model.invoke)(messages).content


def load_dataset() -> list[dict]:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def pick_balanced(entries: list[dict], n_per_topic: int) -> list[dict]:
    """Pick n_per_topic questions from each topic for a representative subset.

    n_per_topic=1 is the smoke test, 2 is the full 10-question dataset.
    """
    selected = []
    for topic in EVAL_TOPICS:
        bucket = [e for e in entries if e.get("topic") == topic]
        selected.extend(bucket[:n_per_topic])
    return selected


def _truncate_contexts(contexts: list[str]) -> list[str]:
    """Trim contexts so total chars stay under MAX_CONTEXT_CHARS.

    Keeps as many full chunks as possible; truncates the last one if needed.
    This keeps the RAGAS judge prompt inside a safe context size.
    """
    result = []
    total = 0
    for chunk in contexts:
        remaining = MAX_CONTEXT_CHARS - total
        if remaining <= 0:
            break
        if len(chunk) <= remaining:
            result.append(chunk)
            total += len(chunk)
        else:
            result.append(chunk[:remaining])
            break
    return result


def _retrieve_for_topic(question: str, topic: str, policy_uin: str) -> list[dict]:
    """Retrieve with the same tool the live agent would pick for this question.

    The agent does not send every question through one generic search: a
    waiting-period question goes to check_waiting_period, an exclusion question
    to check_exclusion, and so on, each filtering Pinecone by topic metadata.
    Evaluating through generic search measured a path the product never uses.
    """
    if topic == "waiting_period":
        return check_waiting_period.invoke(
            {"condition": question, "policy_uin": policy_uin}
        )
    if topic == "exclusion":
        # The bare question ranks general clauses above the exclusion list.
        # Adding the vocabulary an exclusion clause actually uses gives the
        # reranker a stronger signal for the specific clause.
        return check_exclusion.invoke(
            {
                "treatment": f"{question} excluded not covered exclusion clause",
                "policy_uin": policy_uin,
            }
        )
    # coverage, sub_limit and copay_deductible all read payment-condition
    # clauses, and check_coverage already spans those three topics.
    return check_coverage.invoke({"treatment": question, "policy_uin": policy_uin})


def _sql_waiting_period_fact(policy_uin: str, question: str) -> str | None:
    """Return the structured waiting period as a sentence, if one is on record.

    WHY: many policies state the number of months in a table, not in prose --
    the clause text only says "as specified below". That table was routed to
    SQLite at ingest, so the number is genuinely absent from Pinecone. The live
    agent closes this gap with waiting_period_tracker; without it the
    evaluation is scoring a fact the retrieval path was never given.
    """
    for row in get_policy_store().get_waiting_periods(policy_uin):
        condition = (row.get("condition") or "").lower()
        if condition and condition in question.lower():
            return (
                f"Under policy UIN {policy_uin} (page {row.get('page')}), the "
                f"waiting period for {row['condition']} is "
                f"{row['waiting_period_months']} months."
            )
    return None


def answer_question(
    question: str,
    policy_uin: str,
    generator,
    topic: str = "",
) -> tuple[str, list[str]]:
    """Retrieve policy context and generate an answer.

    Retrieval is routed by topic so the evaluation exercises the same tools
    the agent uses, rather than a generic search the product never runs.
    """
    hits = _retrieve_for_topic(question, topic, policy_uin)
    contexts = [hit["text"] for hit in hits]

    if topic == "waiting_period":
        fact = _sql_waiting_period_fact(policy_uin, question)
        if fact:
            # Put it first: it is the exact fact the question asks for.
            contexts.insert(0, fact)

    truncated = _truncate_contexts(contexts)
    context_text = "\n\n".join(truncated)
    answer = generator.answer(question, context_text)
    return answer, truncated


def build_ragas_samples(entries: list[dict]) -> list:
    """Generate answers for every dataset entry."""
    from ragas import SingleTurnSample

    generator = _AnswerGenerator()
    samples = []

    for index, entry in enumerate(entries, start=1):
        print(f"  [{index}/{len(entries)}] {entry['topic']}: {entry['question'][:70]}")
        if index > 1:
            # Groq rate-limits answers generated back to back.
            time.sleep(ANSWER_DELAY_SECONDS)
        answer, contexts = answer_question(
            entry["question"], entry["context_uin"], generator,
            topic=entry.get("topic", ""),
        )
        samples.append(
            SingleTurnSample(
                user_input=entry["question"],
                retrieved_contexts=contexts,
                response=answer,
                reference=entry["ground_truth"],
            )
        )

    return samples


def _make_judge():
    """Build the judge LLM wrapper -- the same Groq model as the answerer."""
    from ragas.llms import LangchainLLMWrapper

    return LangchainLLMWrapper(_groq_llm())


def _score_one(sample, judge_llm, judge_embeddings, metrics, run_config) -> dict:
    """Score a single sample, returning per-metric floats.

    No retry wrapper here on purpose -- run_config already carries
    max_retries/max_wait, and RAGAS retries internally on every judge call.
    Wrapping this in a second retry loop would retry the retries: a single
    stuck question could sit through max_retries-squared attempts before
    finally failing, which is what made an earlier run look hung instead of
    finishing or failing cleanly.
    """
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
    """Score every sample individually, with a pacing delay between questions."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    judge_llm = _make_judge()
    # get_embedder() already returns a LangChain Embeddings, which is what
    # RAGAS wants.
    judge_embeddings = LangchainEmbeddingsWrapper(get_embedder())

    # max_workers=1: serialize calls to avoid hammering Groq's rate limit.
    # timeout=180: retries wait up to RETRY_WAIT_SECONDS; 180s gives that room.
    # max_retries/max_wait reuse the same constants as the answer-generation
    # retry below, so there is one rate-limit policy, not two.
    run_config = RunConfig(
        max_workers=1,
        timeout=180,
        max_retries=MAX_RETRIES,
        max_wait=RETRY_WAIT_SECONDS,
    )
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    accumulated: dict[str, list[float]] = {k: [] for k in METRIC_LABELS}

    for i, sample in enumerate(samples, start=1):
        print(f"\nEvaluating question {i}/{len(samples)}...")
        if i > 1:
            print(f"  Waiting {EVAL_DELAY_SECONDS}s before next call...")
            time.sleep(EVAL_DELAY_SECONDS)

        try:
            scores = _score_one(sample, judge_llm, judge_embeddings, metrics, run_config)
            # Per-question scores make a single bad question visible instead
            # of hiding it inside the average.
            detail = "  ".join(
                f"{label}={scores.get(metric, float('nan')):.2f}"
                for metric, label in METRIC_LABELS.items()
            )
            print(f"  {detail}")
        except Exception as error:
            # A question that cannot be scored is recorded as nan rather than
            # dropped, so the summary still says how many were attempted.
            print(f"  Scoring failed: {error}")
            scores = {}

        for metric in METRIC_LABELS:
            accumulated[metric].append(scores.get(metric, float("nan")))

    def _mean(values: list[float]) -> float:
        valid = [v for v in values if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float("nan")

    return {metric: _mean(accumulated[metric]) for metric in METRIC_LABELS}


def save_results(scores: dict[str, float], dataset_size: int) -> None:
    """Write timestamped scores to baseline_results.json."""
    from config import settings

    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_size": dataset_size,
        "provider": "groq",
        "model": settings.llm_model,
        "note": (
            "Answer and judge are the same model -- scores can run higher "
            "than a different-model judge would give (self-grading bias)."
        ),
        "scores": scores,
    }
    BASELINE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_summary(scores: dict[str, float]) -> None:
    parts = [
        f"{label}: {scores.get(metric, float('nan')):.2f}"
        for metric, label in METRIC_LABELS.items()
    ]
    print("\n" + " | ".join(parts))


def main() -> None:
    import argparse

    from config import settings

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick 5-question smoke test (1 per topic)",
    )
    args = parser.parse_args()

    configure_logging()

    print(f"Model        : {settings.llm_model} (groq)")
    print("Prompt       : eval (strict - context-only)")
    print(f"Context cap  : {MAX_CONTEXT_CHARS:,} chars per question (~3,000 tokens)")
    print(f"Eval delay   : {EVAL_DELAY_SECONDS}s between questions")

    per_topic = 1 if args.smoke else 2
    entries = pick_balanced(load_dataset(), per_topic)
    label = "smoke test" if args.smoke else "full dataset"
    topics = ", ".join(f"{per_topic}x{t}" for t in EVAL_TOPICS)
    print(f"Running the {len(entries)}-question {label} ({topics})")

    print("\nGenerating answers...")
    samples = build_ragas_samples(entries)

    print("\nScoring with RAGAS (faithfulness, relevancy, precision, recall)...")
    scores = run_metrics(samples)

    save_results(scores, len(entries))
    print(f"\nSaved scores   -> {BASELINE_PATH}")
    print_summary(scores)


if __name__ == "__main__":
    main()
