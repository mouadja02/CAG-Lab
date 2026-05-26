"""Config-driven experiment runner.

Loads an experiment YAML, runs every question through the specified architecture,
collects per-question records, and writes a JSONL + report.md to results/.
"""

import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from cag_lab.benchmark.dataset import BenchmarkQuestion, load_dataset
from cag_lab.benchmark.metrics import (
    avg_generation_cost,
    cache_hit_rate,
    citation_present,
    compute_question_cost,
    cost_saved,
    false_positive_hit_rate,
    latency_stats,
    llm_judge_correctness,
    load_pricing,
)


# ---------------------------------------------------------------------------
# Architecture runners
# ---------------------------------------------------------------------------

def _run_classic_rag(config: dict, questions: list[BenchmarkQuestion], pricing: dict) -> list[dict]:
    from cag_lab.generation.answer_generator import generate_answer
    from cag_lab.retrieval.pinecone_retriever import Retriever

    model = config["model"]
    index_name = config["index"]
    top_k = config["top_k"]

    retriever = Retriever(index_name=index_name, top_k=top_k)

    records: list[dict] = []
    for q in questions:
        t0 = time.perf_counter()
        chunks = retriever.retrieve(q.question)
        result = generate_answer(q.question, chunks, model=model)
        latency_s = time.perf_counter() - t0

        judge = llm_judge_correctness(
            q.question,
            q.expected_answer,
            result.answer,
        )

        cost = compute_question_cost(
            result.prompt_tokens, result.completion_tokens, model, pricing
        )

        records.append(
            {
                "question_id": q.id,
                "query_type": q.query_type.value,
                "difficulty": q.difficulty.value,
                "question": q.question,
                "answer": result.answer,
                "cited_sources": result.sources,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": round(latency_s, 4),
                "judge_score": judge["score"],
                "judge_rationale": judge["rationale"],
                "cost": round(cost, 8),
                "has_citation": citation_present(result.answer),
            }
        )

    return records


def _run_semantic_cache_rag(config: dict, workload, pricing: dict) -> list[dict]:
    from cag_lab.cache.semantic_cache import SemanticCache
    from cag_lab.generation.answer_generator import AnswerResult, generate_answer
    from cag_lab.retrieval.pinecone_retriever import Retriever

    model = config["model"]
    index_name = config["index"]
    top_k = config["top_k"]
    threshold = config.get("cache_threshold", 0.92)
    ttl = config.get("cache_ttl_seconds", 3600)

    cache = SemanticCache(similarity_threshold=threshold, ttl_seconds=ttl)
    retriever = Retriever(index_name=index_name, top_k=top_k)

    records: list[dict] = []
    for item in workload:
        t0 = time.perf_counter()
        cache_result = cache.lookup(item.question)
        lookup_latency = time.perf_counter() - t0

        if cache_result.hit:
            false_pos = cache_result.cached_source_id != item.source_id
            gen_cost = compute_question_cost(0, 0, model, pricing)
            avg_cost = _running_avg_gen_cost(records) or 0.0007
            saved = avg_cost - gen_cost

            records.append(
                {
                    "item_id": item.item_id,
                    "question_id": item.source_id,
                    "relationship": item.relationship,
                    "query_type": item.query_type,
                    "difficulty": item.difficulty,
                    "question": item.question,
                    "answer": cache_result.answer,
                    "cited_sources": cache_result.sources,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": round(lookup_latency, 4),
                    "judge_score": 0,
                    "judge_rationale": "",
                    "cost": 0.0,
                    "has_citation": citation_present(cache_result.answer),
                    "cache_hit": True,
                    "cache_match_score": cache_result.score,
                    "cached_question_id": cache_result.cached_source_id,
                    "false_positive": false_pos,
                    "cost_saved": round(saved, 8),
                }
            )

            if not false_pos:
                judge = llm_judge_correctness(
                    item.question,
                    item.expected_answer,
                    cache_result.answer,
                )
                records[-1]["judge_score"] = judge["score"]
                records[-1]["judge_rationale"] = judge["rationale"]
            continue

        t0 = time.perf_counter()
        chunks = retriever.retrieve(item.question)
        result = generate_answer(item.question, chunks, model=model)
        latency_s = time.perf_counter() - t0 + lookup_latency

        judge = llm_judge_correctness(
            item.question,
            item.expected_answer,
            result.answer,
        )

        cost = compute_question_cost(
            result.prompt_tokens, result.completion_tokens, model, pricing
        )

        cache.store(
            query=item.question,
            answer=result.answer,
            sources=result.sources,
            model=model,
            source_id=item.source_id,
        )

        records.append(
            {
                "item_id": item.item_id,
                "question_id": item.source_id,
                "relationship": item.relationship,
                "query_type": item.query_type,
                "difficulty": item.difficulty,
                "question": item.question,
                "answer": result.answer,
                "cited_sources": result.sources,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_s": round(latency_s, 4),
                "judge_score": judge["score"],
                "judge_rationale": judge["rationale"],
                "cost": round(cost, 8),
                "has_citation": citation_present(result.answer),
                "cache_hit": False,
                "cache_match_score": None,
                "cached_question_id": None,
                "false_positive": False,
                "cost_saved": 0.0,
            }
        )

    return records


def _running_avg_gen_cost(records: list[dict]) -> float:
    misses = [r for r in records if not r.get("cache_hit") and r.get("cost", 0) > 0]
    if not misses:
        return 0.0
    return sum(r["cost"] for r in misses) / len(misses)


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

def run_experiment(config_path: str | Path) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config["dataset"])
    base_questions = load_dataset(dataset_path)
    pricing = load_pricing()

    architecture = config["architecture"]
    if architecture == "classic_rag":
        records = _run_classic_rag(config, base_questions, pricing)
        _write_results(config_path.stem, config, records, pricing)
    elif architecture == "semantic_cache_rag":
        from cag_lab.benchmark.workload import generate_workload

        wl_config = config.get("workload", {})
        near_dupe_pairs = config.get("near_duplicate_pairs")
        workload = generate_workload(
            base_questions,
            repeated_query_rate=wl_config.get("repeated_query_rate", 0.4),
            paraphrase_rate=wl_config.get("paraphrase_rate", 0.2),
            new_query_rate=wl_config.get("new_query_rate", 0.4),
            seed=wl_config.get("seed", 42),
            near_duplicate_pairs=near_dupe_pairs,
        )
        records = _run_semantic_cache_rag(config, workload, pricing)
        _write_results(config_path.stem, config, records, pricing)
    elif architecture == "long_context":
        raise NotImplementedError("Architecture 'long_context' is not yet implemented.")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _write_results(experiment_name: str, config: dict, records: list[dict], pricing: dict) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    jsonl_dir = Path("results") / "jsonl"
    reports_dir = Path("results") / "reports"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = jsonl_dir / f"{experiment_name}_{ts}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_report(
        reports_dir / f"{experiment_name}_{ts}_report.md",
        config,
        records,
        pricing,
        jsonl_path.name,
    )


def _write_report(
    report_path: Path,
    config: dict,
    records: list[dict],
    pricing: dict,
    jsonl_name: str,
) -> None:
    model = config["model"]
    architecture = config["architecture"]
    n = len(records)
    latencies = [r["latency_s"] for r in records]
    l_stats = latency_stats(latencies)

    scores = [r["judge_score"] for r in records]
    mean_score = sum(scores) / n if n else 0.0

    cite_count = sum(1 for r in records if r["has_citation"])
    cite_rate = cite_count / n if n else 0.0

    total_cost = sum(r["cost"] for r in records)
    cost_per_1k = (total_cost / n * 1000) if n else 0.0

    total_prompt_tokens = sum(r["prompt_tokens"] for r in records)
    total_completion_tokens = sum(r["completion_tokens"] for r in records)

    is_cache = any(r.get("cache_hit") is not None for r in records)

    type_groups: dict[str, list[dict]] = {}
    for r in records:
        type_groups.setdefault(r["query_type"], []).append(r)

    lines = [
        f"# {config_path_to_title(config)}",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Results file:** {jsonl_name}",
        "",
        "## Configuration",
        "",
        f"| Key | Value |",
        f"|-----|-------|",
        f"| Architecture | {architecture} |",
        f"| Dataset | {config['dataset']} |",
        f"| Index | {config['index']} |",
        f"| Model | {model} |",
        f"| Top-K | {config['top_k']} |",
        f"| Questions | {n} |",
    ]

    if is_cache:
        wl = config.get("workload", {})
        lines += [
            f"| Cache threshold | {config.get('cache_threshold', 0.92)} |",
            f"| Cache TTL | {config.get('cache_ttl_seconds', 3600)} s |",
            f"| Repeat rate | {wl.get('repeated_query_rate', '-')} |",
            f"| Paraphrase rate | {wl.get('paraphrase_rate', '-')} |",
            f"| New rate | {wl.get('new_query_rate', '-')} |",
        ]

    lines += [
        "",
        "## Aggregate Quality",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean judge score | {mean_score:.3f} |",
        f"| Citation rate | {cite_rate:.1%} |",
        f"| p50 latency | {l_stats['p50']:.2f} s |",
        f"| p95 latency | {l_stats['p95']:.2f} s |",
        f"| Total tokens | {total_prompt_tokens + total_completion_tokens:,} ({total_prompt_tokens:,} prompt + {total_completion_tokens:,} completion) |",
        f"| Total cost | ${total_cost:.6f} |",
        f"| Cost per 1k questions | ${cost_per_1k:.6f} |",
    ]

    if is_cache:
        hit_rate = cache_hit_rate(records)
        fp_rate = false_positive_hit_rate(records)
        saved = cost_saved(records)
        rel_groups: dict[str, list[dict]] = {}
        for r in records:
            rel_groups.setdefault(r.get("relationship", "new"), []).append(r)

        lines += [
            "",
            "## Cache Performance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Cache hit rate | {hit_rate:.1%} |",
            f"| False-positive hit rate | {fp_rate:.1%} |",
            f"| Cost saved | ${saved:.6f} |",
            "",
            "### Hit rate by relationship",
            "",
            "| Relationship | Count | Hit Rate |",
            "|--------------|-------|----------|",
        ]
        for rel in ["exact", "paraphrase", "new"]:
            group = rel_groups.get(rel, [])
            g_n = len(group)
            g_hits = sum(1 for r in group if r.get("cache_hit"))
            g_rate = g_hits / g_n if g_n else 0.0
            lines.append(f"| {rel} | {g_n} | {g_rate:.1%} |")

    lines += [
        "",
        "## Per Query Type",
        "",
        "| Query Type | Count | Mean Score | Citation Rate |",
        "|------------|-------|------------|---------------|",
    ]

    for qtype in sorted(type_groups):
        group = type_groups[qtype]
        g_n = len(group)
        g_score = sum(r["judge_score"] for r in group) / g_n if g_n else 0.0
        g_cite = sum(1 for r in group if r["has_citation"]) / g_n if g_n else 0.0
        lines.append(f"| {qtype} | {g_n} | {g_score:.3f} | {g_cite:.1%} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def config_path_to_title(config: dict) -> str:
    arch = config.get("architecture", "unknown")
    model = config.get("model", "?")
    top_k = config.get("top_k", "?")
    if arch == "semantic_cache_rag":
        threshold = config.get("cache_threshold", 0.92)
        return f"Semantic-Cache RAG — {model} @ top-{top_k} (threshold={threshold})"
    return f"RAG Baseline — {model} @ top-{top_k}"
