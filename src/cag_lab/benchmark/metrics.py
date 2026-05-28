"""Per-question metrics: latency, cost, correctness, and citation presence.

Each function is small and standalone.
"""

import math
import re
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def latency_stats(latencies_seconds: list[float]) -> dict:
    """Compute latency distribution from a list of wall-clock latencies (s)."""
    if not latencies_seconds:
        return {
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    sorted_l = sorted(latencies_seconds)
    n = len(sorted_l)

    def _pctile(pct: float) -> float:
        k = (pct / 100.0) * (n - 1)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_l[int(k)]
        return sorted_l[f] * (c - k) + sorted_l[c] * (k - f)

    return {
        "p25": _pctile(25),
        "p50": _pctile(50),
        "p75": _pctile(75),
        "p95": _pctile(95),
        "p99": _pctile(99),
        "min": sorted_l[0],
        "max": sorted_l[-1],
        "mean": sum(sorted_l) / n,
    }


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def load_pricing(pricing_path: str | Path = "configs/models/pricing.yaml") -> dict:
    """Load model pricing from YAML file."""
    with open(pricing_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_question_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    pricing: dict,
) -> float:
    """Cost for one question given token counts and the pricing table."""
    model_pricing = pricing.get(model)
    if model_pricing is None:
        from warnings import warn

        warn(
            f"Model '{model}' not found in pricing.yaml — cost set to $0", stacklevel=2
        )
        model_pricing = {}
    input_price = model_pricing.get("input", 0.0)
    output_price = model_pricing.get("output", 0.0)
    cost = (prompt_tokens / 1_000_000) * input_price + (
        completion_tokens / 1_000_000
    ) * output_price
    return cost


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator. Compare the generated answer to the expected answer "
    "and decide if the generated answer is CORRECT (score=1) or INCORRECT (score=0). "
    "A correct answer conveys the same essential information as the expected answer, "
    "even if wording differs. An incorrect answer is missing key facts, contradicts "
    "the expected answer, or adds incorrect information.\n\n"
    "Respond with a JSON object with two keys:\n"
    '  "score": 0 or 1\n'
    '  "rationale": one short sentence explaining your decision'
)


def llm_judge_correctness(
    question: str,
    expected_answer: str,
    generated_answer: str,
    *,
    judge_model: str,
    api_base: str | None = None,
) -> dict:
    """LLM-as-judge correctness score (0 or 1) with a one-line rationale.

    **Costs money per question** — each call uses the judge_model LLM.
    """
    from cag_lab.generation.llm_client import complete

    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Expected answer: {expected_answer}\n\n"
                f"Generated answer: {generated_answer}"
            ),
        },
    ]
    result = complete(judge_model, messages, api_base=api_base)

    import json

    try:
        parsed = json.loads(result.answer)
        return {
            "score": int(parsed.get("score", 0)),
            "rationale": str(parsed.get("rationale", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0, "rationale": "judge output unparseable"}


# ---------------------------------------------------------------------------
# Retrieval relevance
# ---------------------------------------------------------------------------

_RELEVANCE_SYSTEM_PROMPT = (
    "You are a retrieval-quality evaluator. Given a question, an expected answer, "
    "and a set of retrieved document chunks, judge whether the chunks contain "
    "enough information to answer the question correctly.\n\n"
    "Score from 0 to 2:\n"
    "  0 = chunks are irrelevant or missing critical information\n"
    "  1 = chunks contain partial information but not enough for a complete answer\n"
    "  2 = chunks contain all information needed to produce the expected answer\n\n"
    "Respond with a JSON object with two keys:\n"
    '  "score": 0, 1, or 2\n'
    '  "rationale": one short sentence explaining your decision'
)


def retrieval_relevance(
    question: str,
    expected_answer: str,
    retrieved_chunks: list[str],
    *,
    judge_model: str,
    api_base: str | None = None,
) -> dict:
    """Score whether retrieved chunks contain the information needed to answer.

    Returns {"score": 0|1|2, "rationale": str}.
    **Costs money per call** — uses the judge_model LLM.
    """
    from cag_lab.generation.llm_client import complete

    chunks_text = "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(retrieved_chunks, 1)
    )
    messages = [
        {"role": "system", "content": _RELEVANCE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Expected answer: {expected_answer}\n\n"
                f"Retrieved chunks:\n{chunks_text}"
            ),
        },
    ]
    result = complete(judge_model, messages, api_base=api_base)

    import json

    try:
        parsed = json.loads(result.answer)
        return {
            "score": int(parsed.get("score", 0)),
            "rationale": str(parsed.get("rationale", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": 0, "rationale": "relevance judge output unparseable"}


# ---------------------------------------------------------------------------
# Citation presence
# ---------------------------------------------------------------------------


def citation_present(answer: str) -> bool:
    """Return True if the answer text includes a Sources line."""
    return bool(re.search(r"Sources:", answer, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Cache metrics
# ---------------------------------------------------------------------------


def cache_hit_rate(records: list[dict]) -> float:
    """Fraction of questions served from cache."""
    if not records:
        return 0.0
    hits = sum(1 for r in records if r.get("cache_hit"))
    return hits / len(records)


def false_positive_hit_rate(records: list[dict]) -> float:
    """Fraction of cache hits that were served for a *different* question
    (cached source_id != query source_id)."""
    hits = [r for r in records if r.get("cache_hit")]
    if not hits:
        return 0.0
    fp = sum(1 for r in hits if r.get("false_positive"))
    return fp / len(hits)


def cost_saved(records: list[dict]) -> float:
    """Total generation cost saved by serving answers from cache."""
    return sum(r.get("cost_saved", 0.0) for r in records)


def avg_generation_cost(records: list[dict]) -> float:
    """Average generation cost for cache misses (full RAG answers)."""
    misses = [r for r in records if not r.get("cache_hit") and r.get("cost", 0) > 0]
    if not misses:
        return 0.0
    return sum(r["cost"] for r in misses) / len(misses)
