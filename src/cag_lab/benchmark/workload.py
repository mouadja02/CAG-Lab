"""Workload generator for cache benchmarking.

Takes a base dataset and emits an ordered workload with ground-truth labels
(exact repeat / paraphrase / new) so the cache can be scored honestly.
"""

import math
import random
from dataclasses import dataclass

from cag_lab.benchmark.dataset import BenchmarkQuestion


@dataclass
class WorkloadItem:
    item_id: str
    question: str
    source_id: str
    relationship: str  # "exact", "paraphrase", or "new"
    expected_answer: str
    query_type: str
    difficulty: str
    paraphrase_tier: str | None = (
        None  # "easy", "medium", "hard" (only for paraphrases)
    )


def generate_workload(
    dataset: list[BenchmarkQuestion],
    repeated_query_rate: float = 0.3,
    paraphrase_rate: float = 0.2,
    new_query_rate: float = 0.5,
    seed: int = 42,
    near_duplicate_pairs: list[dict] | None = None,
    paraphrase_mode: str = "template",
    paraphrase_model: str = "gpt-4o-mini",
) -> list[WorkloadItem]:
    """Generate a synthetic workload from base questions.

    Args:
        paraphrase_mode: "template" for cheap prefix-based paraphrases (original),
                         "llm" for LLM-generated paraphrases with tiered difficulty.
        paraphrase_model: Model to use for LLM paraphrase generation.
    """
    total_rate = repeated_query_rate + paraphrase_rate + new_query_rate
    if abs(total_rate - 1.0) > 0.001:
        raise ValueError(
            f"Rates must sum to 1.0, got {repeated_query_rate}+{paraphrase_rate}+{new_query_rate}={total_rate}"
        )

    rng = random.Random(seed)
    base_n = len(dataset)
    total = math.ceil(base_n / new_query_rate) if new_query_rate > 0 else base_n
    n_repeat = max(0, round(total * repeated_query_rate))
    n_paraphrase = max(0, round(total * paraphrase_rate))

    emitted_new: list[WorkloadItem] = []
    counter = 0

    def _item_id() -> str:
        nonlocal counter
        counter += 1
        return f"wl-{counter:04d}"

    for q in dataset:
        item = WorkloadItem(
            item_id=_item_id(),
            question=q.question,
            source_id=q.id,
            relationship="new",
            expected_answer=q.expected_answer,
            query_type=q.query_type.value,
            difficulty=q.difficulty.value,
        )
        emitted_new.append(item)

    workload = list(emitted_new)

    repeats = rng.choices(emitted_new, k=n_repeat)
    for ref in repeats:
        workload.append(
            WorkloadItem(
                item_id=_item_id(),
                question=ref.question,
                source_id=ref.source_id,
                relationship="exact",
                expected_answer=ref.expected_answer,
                query_type=ref.query_type,
                difficulty=ref.difficulty,
            )
        )

    paraphrases = rng.choices(emitted_new, k=n_paraphrase)

    if paraphrase_mode == "llm":
        paraphrased_texts = _generate_llm_paraphrases(
            [ref.question for ref in paraphrases], paraphrase_model
        )
        for ref, (text, tier) in zip(paraphrases, paraphrased_texts):
            workload.append(
                WorkloadItem(
                    item_id=_item_id(),
                    question=text,
                    source_id=ref.source_id,
                    relationship="paraphrase",
                    expected_answer=ref.expected_answer,
                    query_type=ref.query_type,
                    difficulty=ref.difficulty,
                    paraphrase_tier=tier,
                )
            )
    else:
        for ref in paraphrases:
            workload.append(
                WorkloadItem(
                    item_id=_item_id(),
                    question=_paraphrase(ref.question, rng),
                    source_id=ref.source_id,
                    relationship="paraphrase",
                    expected_answer=ref.expected_answer,
                    query_type=ref.query_type,
                    difficulty=ref.difficulty,
                    paraphrase_tier="easy",
                )
            )

    if near_duplicate_pairs:
        for pair in near_duplicate_pairs:
            workload.append(
                WorkloadItem(
                    item_id=_item_id(),
                    question=pair["question"],
                    source_id=pair["source_id"],
                    relationship="new",
                    expected_answer=pair.get("expected_answer", ""),
                    query_type=pair.get("query_type", "factual"),
                    difficulty=pair.get("difficulty", "medium"),
                )
            )

    groups: dict[str, list[WorkloadItem]] = {}
    for item in workload:
        groups.setdefault(item.source_id, []).append(item)

    for gid in groups:
        groups[gid].sort(
            key=lambda i: {"new": 0, "paraphrase": 1, "exact": 2}[i.relationship]
        )

    sorted_workload: list[WorkloadItem] = []
    group_order = list(groups.keys())
    rng.shuffle(group_order)
    for gid in group_order:
        sorted_workload.extend(groups[gid])

    return sorted_workload


# ---------------------------------------------------------------------------
# Template-based paraphrasing (cheap, shallow)
# ---------------------------------------------------------------------------

_PARAPHRASE_PREFIXES = [
    "Can you tell me: {}",
    "Please explain: {}",
    "I'd like to understand: {}",
    "Could you clarify: {}",
    "Tell me about: {}",
    "What is your take on: {}",
    "Elaborate on: {}",
    "Explain to me: {}",
]


def _paraphrase(text: str, rng: random.Random) -> str:
    return rng.choice(_PARAPHRASE_PREFIXES).format(text)


# ---------------------------------------------------------------------------
# LLM-based paraphrasing (realistic, tiered difficulty)
# ---------------------------------------------------------------------------

_PARAPHRASE_SYSTEM_PROMPT = """\
You are a paraphrase generator. Given a question, produce exactly 3 rephrasings \
at different difficulty tiers. Each rephrasing must ask the SAME question but \
with different wording.

Tiers:
- easy: Swap a few synonyms or reorder clauses. Keep most original words.
- medium: Rewrite the sentence structure (passive voice, nominalization, split \
into sub-questions then merge). Change terminology where possible.
- hard: Completely rephrase using different framing, analogies, or indirect \
phrasing. A reader should need to think to realize it's the same question.

Respond with a JSON array of 3 objects, each with "tier" and "text" keys.
Example: [{"tier":"easy","text":"..."},{"tier":"medium","text":"..."},{"tier":"hard","text":"..."}]"""


def _generate_llm_paraphrases(
    questions: list[str], model: str
) -> list[tuple[str, str]]:
    """Generate tiered paraphrases for a batch of questions.

    Returns a list of (paraphrased_text, tier) tuples, one per input question.
    Each question gets a randomly selected tier from the 3 generated options,
    weighted toward harder tiers to stress-test the cache.
    """
    import json
    import random as stdlib_random

    from cag_lab.generation.llm_client import complete
    from tqdm import tqdm

    results: list[tuple[str, str]] = []
    tier_weights = {"easy": 1, "medium": 2, "hard": 3}  # bias toward harder

    for question in tqdm(questions, desc="  Paraphrasing ", unit="q"):
        messages = [
            {"role": "system", "content": _PARAPHRASE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        try:
            result = complete(model, messages)
            parsed = json.loads(result.answer)
            if isinstance(parsed, list) and len(parsed) >= 3:
                # Weighted random selection biased toward harder tiers
                weights = [tier_weights.get(p.get("tier", "easy"), 1) for p in parsed]
                chosen = stdlib_random.choices(parsed, weights=weights, k=1)[0]
                results.append((chosen["text"], chosen["tier"]))
            else:
                # Fallback: use first item or raw text
                results.append((parsed[0]["text"] if parsed else question, "easy"))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            # Fallback to template paraphrase on failure
            rng = stdlib_random.Random()
            results.append((_paraphrase(question, rng), "easy"))

    return results
