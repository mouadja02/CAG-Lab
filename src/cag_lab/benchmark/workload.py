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


def generate_workload(
    dataset: list[BenchmarkQuestion],
    repeated_query_rate: float = 0.3,
    paraphrase_rate: float = 0.2,
    new_query_rate: float = 0.5,
    seed: int = 42,
    near_duplicate_pairs: list[dict] | None = None,
) -> list[WorkloadItem]:
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

    rng.shuffle(workload)
    return workload


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
