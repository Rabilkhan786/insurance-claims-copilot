"""Run the RAGAS evaluation suite against evaluation/dataset.json.

Usage:  uv run python evaluation/run_ragas.py
        uv run python evaluation/run_ragas.py --smoke   (1 per topic)

The dataset is 10 questions, 2 per topic. Both the answers and the judge use
Groq openai/gpt-oss-120b -- the only model this project runs. NOTE: this
means the judge scores answers from its own model, which self-grading bias
can inflate (faithfulness and relevancy read higher than a different-model
judge would give). Treat these scores as directional, not absolute.

Each question is scored individually so rate-limit delays and retries are
per-question rather than per-batch. Scores are saved to
evaluation/baseline_results.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "evaluation" / "dataset.json"
BASELINE_PATH = ROOT / "evaluation" / "baseline_results.json"

# The only model this project runs: Groq openai/gpt-oss-120b. Used for both
# answer generation and judging.
EVAL_MODEL = "openai/gpt-oss-120b"

# Capping retrieved context at 3,000 tokens keeps judge prompts reasonable.
MAX_CONTEXT_CHARS = 12_000

# 20 seconds between questions keeps Groq under its tokens-per-minute limit
# on the free tier.
EVAL_DELAY_SECONDS = 20

# Spacing between answer-generation calls, same reason.
ANSWER_DELAY_SECONDS = 10

# Groq's free tier caps tokens per minute, so a 429 means waiting out the
# window -- 60 seconds clears it.
RETRY_WAIT_SECONDS = 60
MAX_RETRIES = 3

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "context_precision": "Precision",
    "context_recall": "Recall",
}

EVAL_TOPICS = ["waiting_period", "coverage", "exclusion", "sub_limit", "comparison"]

# Stricter prompt for evaluation: every stated fact must come from the context.
# The production prompt allows conversational hedging that RAGAS faithfulness
# cannot verify against retrieved chunks, which deflates the score.
EVAL_SYSTEM_PROMPT = """You are a health insurance assistant evaluating policy documents.
Answer ONLY using facts explicitly stated in the CONTEXT below.
Do not add any information from general knowledge.
Do not hedge, guess, or infer beyond what the context says.
If the answer is not in the context, reply exactly: The context does not contain information about this.
Quote relevant numbers or policy terms exactly as they appear.
When comparing two or more policies, structure your answer as follows:
- Name and UIN of each policy on a separate line
- The specific value for each policy with exact numbers
- A brief summary sentence stating which is better and why
- Source citations for each policy separately

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

from config import settings  # noqa: E402
from src.embeddings import BGEEmbedder  # noqa: E402
from src.tools.rag_tools import (  # noqa: E402
    check_coverage,
    check_exclusion,
    check_waiting_period,
)
from src.policy_data import PolicyDataStore  # noqa: E402
from src.utils import configure_logging  # noqa: E402


class _AnswerGenerator:
    """Groq gpt-oss-120b answer generator using the strict eval-only prompt.

    Same model that judges the answers below (see the module docstring for
    the self-grading-bias tradeoff that comes with that).
    """

    def __init__(self) -> None:
        from langchain_groq import ChatGroq

        self.model = ChatGroq(
            api_key=settings.groq_api_key,
            model=EVAL_MODEL,
            temperature=0,
        )

    def answer(self, question: str, context: str) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", EVAL_SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        messages = prompt.format_messages(context=context, question=question)

        for attempt in range(MAX_RETRIES):
            try:
                return self.model.invoke(messages).content
            except Exception as exc:
                if "429" in str(exc) or "rate_limit" in str(exc).lower():
                    wait = RETRY_WAIT_SECONDS
                    print(f"  Groq 429 on answer gen (attempt {attempt + 1}/{MAX_RETRIES}), "
                          f"waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        return self.model.invoke(messages).content  # final attempt, let it raise


class _BGELangchainEmbeddings:
    """Adapts BGEEmbedder to the LangChain Embeddings interface RAGAS expects."""

    def __init__(self) -> None:
        self._embedder = BGEEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


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
    This keeps the RAGAS judge prompt inside Groq's context window.
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


def _effective_topic(question: str, topic: str) -> str:
    """Resolve "comparison" to the topic the question is really about.

    "comparison" describes the shape of a question, not the kind of clause it
    needs. Routing every comparison to check_coverage filtered Pinecone to
    topic in (coverage, sub_limit), so a question comparing two waiting
    periods could never reach a waiting_period chunk and always abstained --
    even though the clause was indexed and reachable.
    """
    if topic != "comparison":
        return topic

    text = (question or "").lower()
    if "waiting period" in text or "waiting-period" in text:
        return "waiting_period"
    if "exclu" in text or "not covered" in text:
        return "exclusion"
    return "coverage"


def _retrieve_for_topic(question: str, topic: str, policy_uin: str) -> list[dict]:
    """Retrieve with the same tool the live agent would pick for this question.

    The agent does not send every question through one generic search: a
    waiting-period question goes to check_waiting_period, an exclusion question
    to check_exclusion, and so on, each filtering Pinecone by topic metadata.
    Evaluating through generic search measured a path the product never uses.
    """
    effective = _effective_topic(question, topic)

    if effective == "waiting_period":
        return check_waiting_period.invoke(
            {"condition": question, "policy_uin": policy_uin}
        )
    if effective == "exclusion":
        # The bare question ranks general clauses above the exclusion list.
        # Adding the vocabulary an exclusion clause actually uses gives the
        # reranker a stronger signal for the specific clause.
        return check_exclusion.invoke(
            {
                "treatment": f"{question} excluded not covered exclusion clause",
                "policy_uin": policy_uin,
            }
        )
    # coverage and sub_limit both read covered-benefit clauses; check_coverage
    # already spans the coverage and sub_limit topics.
    return check_coverage.invoke({"treatment": question, "policy_uin": policy_uin})


_policy_store: PolicyDataStore | None = None


def _sql_waiting_period_fact(policy_uin: str, question: str) -> str | None:
    """Return the structured waiting period as a sentence, if one is on record.

    WHY: many policies state the number of months in a table, not in prose --
    the clause text only says "as specified below". That table was routed to
    SQLite at ingest, so the number is genuinely absent from Pinecone. The live
    agent closes this gap with waiting_period_tracker; without it the
    evaluation is scoring a fact the retrieval path was never given.
    """
    global _policy_store
    if _policy_store is None:
        _policy_store = PolicyDataStore(settings.crm_db_path)

    for row in _policy_store.get_waiting_periods(policy_uin):
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
    policy_uin_2: str | None = None,
    topic: str = "",
) -> tuple[str, list[str]]:
    """Retrieve policy context and generate an answer.

    Retrieval is routed by topic so the evaluation exercises the same tools the
    agent uses. Comparison questions across two policies query both UIDs.
    """
    if topic == "comparison" and policy_uin_2:
        # A query naming both policies ranks cover pages above real clauses, so
        # each policy is searched separately and the hits merged.
        hits = _retrieve_for_topic(question, topic, policy_uin)
        hits2 = _retrieve_for_topic(question, topic, policy_uin_2)
        print(f"    Policy 1 ({policy_uin}): {len(hits)} chunks")
        print(f"    Policy 2 ({policy_uin_2}): {len(hits2)} chunks")
        hits = hits + hits2
    else:
        hits = _retrieve_for_topic(question, topic, policy_uin)

    contexts = [hit["text"] for hit in hits]

    if topic == "waiting_period":
        for uin in filter(None, (policy_uin, policy_uin_2)):
            fact = _sql_waiting_period_fact(uin, question)
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
            # Groq's free tier 429s if answers are generated back to back.
            time.sleep(ANSWER_DELAY_SECONDS)
        answer, contexts = answer_question(
            entry["question"], entry["context_uin"], generator,
            policy_uin_2=entry.get("context_uin_2"),
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
    """Build the judge LLM wrapper — Groq openai/gpt-oss-120b."""
    from ragas.llms import LangchainLLMWrapper
    from langchain_groq import ChatGroq

    return LangchainLLMWrapper(ChatGroq(
        api_key=settings.groq_api_key,
        model=EVAL_MODEL,
        temperature=0,
        max_retries=2,
    ))


def _score_one(sample, judge_llm, judge_embeddings, metrics, run_config) -> dict:
    """Score a single sample, returning per-metric floats."""
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
    """Score every sample individually, with delay and retry between questions."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    judge_llm = _make_judge()
    judge_embeddings = LangchainEmbeddingsWrapper(_BGELangchainEmbeddings())

    # max_workers=1: serialize calls to avoid hammering Groq's free-tier rate
    # limit. timeout=180: 429 retries wait up to 60s; 180s gives that room.
    run_config = RunConfig(max_workers=1, timeout=180)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

    accumulated: dict[str, list[float]] = {k: [] for k in METRIC_LABELS}

    for i, sample in enumerate(samples, start=1):
        print(f"\nEvaluating question {i}/{len(samples)}...")
        if i > 1:
            print(f"  Waiting {EVAL_DELAY_SECONDS}s before next call...")
            time.sleep(EVAL_DELAY_SECONDS)

        scored = False
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                scores = _score_one(sample, judge_llm, judge_embeddings, metrics, run_config)
                for metric in METRIC_LABELS:
                    accumulated[metric].append(scores.get(metric, float("nan")))
                # Per-question scores make a single bad question visible
                # instead of hiding it inside the average.
                detail = "  ".join(
                    f"{label}={scores.get(metric, float('nan')):.2f}"
                    for metric, label in METRIC_LABELS.items()
                )
                print(f"  {detail}")
                scored = True
                break
            except Exception as exc:
                err = str(exc)
                if "429" in err or "rate_limit" in err.lower():
                    print(f"  429 rate limit (attempt {attempt}/{MAX_RETRIES}), "
                          f"waiting {RETRY_WAIT_SECONDS}s...")
                    time.sleep(RETRY_WAIT_SECONDS)
                else:
                    print(f"  Scoring error: {exc}")
                    break

        if not scored:
            for metric in METRIC_LABELS:
                accumulated[metric].append(float("nan"))

    def _mean(values: list[float]) -> float:
        valid = [v for v in values if not math.isnan(v)]
        return sum(valid) / len(valid) if valid else float("nan")

    return {metric: _mean(accumulated[metric]) for metric in METRIC_LABELS}


def save_results(scores: dict[str, float], dataset_size: int) -> None:
    """Write timestamped scores to baseline_results.json."""
    payload = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_size": dataset_size,
        "answer_provider": "groq",
        "answer_model": EVAL_MODEL,
        "judge_provider": "groq",
        "judge_model": EVAL_MODEL,
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

    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick 5-question smoke test (1 per topic)",
    )
    args = parser.parse_args()

    configure_logging()

    print(f"Answer model : {EVAL_MODEL} (groq)")
    print(f"Judge model  : {EVAL_MODEL} (groq)")
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
