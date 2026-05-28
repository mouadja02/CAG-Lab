"""Config-driven experiment runner.

Loads an experiment YAML, runs every question through the specified architecture,
collects per-question records, and writes a JSONL + report.md to results/.
"""

import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

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
    retrieval_relevance,
)

_REPORT_INTERVAL = 10
_RESUME_MAX_AGE_HOURS = 6


# ---------------------------------------------------------------------------
# Architecture runners
# ---------------------------------------------------------------------------


def _run_classic_rag(
    config: dict,
    questions: list[BenchmarkQuestion],
    pricing: dict,
    jsonl_path: Path,
    report_path: Path,
    resumed_ids: set[str],
    existing_records: list[dict],
) -> list[dict]:
    from cag_lab.generation.answer_generator import generate_answer
    from cag_lab.retrieval import Retriever

    model = config["model"]
    index_name = config["index"]
    top_k = config["top_k"]
    judge_model = config.get("judge_model")
    eval_retrieval = config.get("eval_retrieval", False)

    retriever = Retriever(index_name=index_name, top_k=top_k)

    records: list[dict] = list(existing_records)
    jsonl_dir = jsonl_path.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for q in tqdm(questions, desc="  Classic RAG  ", unit="q"):
            if q.id in resumed_ids:
                continue

            t0 = time.perf_counter()
            chunks = retriever.retrieve(q.question)
            result = generate_answer(q.question, chunks, model=model)
            latency_s = time.perf_counter() - t0

            judge = llm_judge_correctness(
                q.question,
                q.expected_answer,
                result.answer,
                judge_model=judge_model,
            )

            cost = compute_question_cost(
                result.prompt_tokens, result.completion_tokens, model, pricing
            )

            record = {
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

            if eval_retrieval:
                chunk_texts = [c.text for c in chunks]
                rel = retrieval_relevance(
                    q.question,
                    q.expected_answer,
                    chunk_texts,
                    judge_model=judge_model,
                )
                record["retrieval_relevance"] = rel["score"]
                record["retrieval_rationale"] = rel["rationale"]

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            records.append(record)

            if len(records) % _REPORT_INTERVAL == 0:
                _write_report(report_path, config, records, pricing, jsonl_path.name)

    return records


def _run_semantic_cache_rag(
    config: dict,
    workload,
    pricing: dict,
    jsonl_path: Path,
    report_path: Path,
    resumed_ids: set[str],
    existing_records: list[dict],
) -> list[dict]:
    from cag_lab.cache.semantic_cache import SemanticCache
    from cag_lab.generation.answer_generator import AnswerResult, generate_answer
    from cag_lab.retrieval import Retriever

    model = config["model"]
    index_name = config["index"]
    top_k = config["top_k"]
    threshold = config.get("cache_threshold", 0.92)
    ttl = config.get("cache_ttl_seconds", 3600)
    judge_model = config.get("judge_model")

    cache = SemanticCache(similarity_threshold=threshold, ttl_seconds=ttl)
    retriever = Retriever(index_name=index_name, top_k=top_k)

    records: list[dict] = list(existing_records)
    jsonl_dir = jsonl_path.parent
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for item in tqdm(workload, desc="  Cache RAG    ", unit="q"):
            if item.item_id in resumed_ids:
                continue

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
                        "paraphrase_tier": item.paraphrase_tier,
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
                        judge_model=judge_model,
                    )
                    records[-1]["judge_score"] = judge["score"]
                    records[-1]["judge_rationale"] = judge["rationale"]

                f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
                f.flush()

                if len(records) % _REPORT_INTERVAL == 0:
                    _write_report(
                        report_path, config, records, pricing, jsonl_path.name
                    )
                continue

            t0 = time.perf_counter()
            chunks = retriever.retrieve(item.question)
            result = generate_answer(item.question, chunks, model=model)
            latency_s = time.perf_counter() - t0 + lookup_latency

            judge = llm_judge_correctness(
                item.question,
                item.expected_answer,
                result.answer,
                judge_model=judge_model,
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
                    "paraphrase_tier": item.paraphrase_tier,
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

            f.write(json.dumps(records[-1], ensure_ascii=False) + "\n")
            f.flush()

            if len(records) % _REPORT_INTERVAL == 0:
                _write_report(report_path, config, records, pricing, jsonl_path.name)

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

    llm_model = os.getenv("LLM_MODEL")
    if llm_model:
        config["model"] = llm_model

    judge_model = os.getenv("JUDGE_MODEL")
    if judge_model:
        config["judge_model"] = judge_model

    dataset_path = Path(config["dataset"])
    base_questions = load_dataset(dataset_path)
    pricing = load_pricing()

    architecture = config["architecture"]
    model = config.get("model", "?")
    print(f"\n{'=' * 60}")
    print(f"  Experiment: {architecture} | model={model}")
    print(f"{'=' * 60}")

    jsonl_path, report_path, resumed_ids, existing_records = _prepare_output(
        config_path.stem
    )

    if resumed_ids:
        print(f"  Resuming: {len(resumed_ids)} already completed")

    if architecture == "classic_rag":
        print(
            f"  Questions: {len(base_questions)} | index={config.get('index')} | top_k={config.get('top_k')}"
        )
        records = _run_classic_rag(
            config,
            base_questions,
            pricing,
            jsonl_path,
            report_path,
            resumed_ids,
            existing_records,
        )
        _write_report(report_path, config, records, pricing, jsonl_path.name)
    elif architecture == "semantic_cache_rag":
        from cag_lab.benchmark.workload import generate_workload

        wl_config = config.get("workload", {})
        near_dupe_pairs = config.get("near_duplicate_pairs")
        print(
            f"  Generating workload (paraphrase_mode={wl_config.get('paraphrase_mode', 'template')})..."
        )
        workload = generate_workload(
            base_questions,
            repeated_query_rate=wl_config.get("repeated_query_rate", 0.4),
            paraphrase_rate=wl_config.get("paraphrase_rate", 0.2),
            new_query_rate=wl_config.get("new_query_rate", 0.4),
            seed=wl_config.get("seed", 42),
            near_duplicate_pairs=near_dupe_pairs,
            paraphrase_mode=wl_config.get("paraphrase_mode", "template"),
            paraphrase_model=wl_config.get("paraphrase_model", "gpt-4o-mini"),
        )
        print(
            f"  Workload: {len(workload)} items | threshold={config.get('cache_threshold', 0.92)}"
        )
        records = _run_semantic_cache_rag(
            config,
            workload,
            pricing,
            jsonl_path,
            report_path,
            resumed_ids,
            existing_records,
        )
        _write_report(report_path, config, records, pricing, jsonl_path.name)
    elif architecture == "long_context":
        raise NotImplementedError("Architecture 'long_context' is not yet implemented.")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _prepare_output(experiment_name: str) -> tuple[Path, Path, set[str], list[dict]]:
    """Create output paths and detect resume state.

    Returns (jsonl_path, report_path, resumed_ids, existing_records).
    If an in-progress file exists within _RESUME_MAX_AGE_HOURS, reuse it and
    return the set of already-completed IDs + loaded records.
    """
    jsonl_dir = Path("results") / "jsonl"
    reports_dir = Path("results") / "reports"
    jsonl_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    candidates = sorted(
        jsonl_dir.glob(f"{experiment_name}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        latest = candidates[0]
        age_hours = (time.time() - latest.stat().st_mtime) / 3600
        if 0 < age_hours <= _RESUME_MAX_AGE_HOURS:
            resumed_ids: set[str] = set()
            existing_records: list[dict] = []
            with open(latest, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        existing_records.append(rec)
                        qid = rec.get("item_id") or rec.get("question_id")
                        if qid:
                            resumed_ids.add(qid)
                    except json.JSONDecodeError:
                        pass
            jsonl_path = latest
            report_path = reports_dir / f"{jsonl_path.stem}_report.md"
            return jsonl_path, report_path, resumed_ids, existing_records

    jsonl_path = jsonl_dir / f"{experiment_name}_{ts}.jsonl"
    report_path = reports_dir / f"{experiment_name}_{ts}_report.md"
    return jsonl_path, report_path, set(), []


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
    latencies = [r["latency_s"] for r in records if r]
    l_stats = latency_stats(latencies)

    scores = [r["judge_score"] for r in records if r]
    mean_score = sum(scores) / n if n else 0.0

    cite_count = sum(1 for r in records if r and r.get("has_citation"))
    cite_rate = cite_count / n if n else 0.0

    total_cost = sum(r.get("cost", 0.0) for r in records)
    cost_per_1k = (total_cost / n * 1000) if n else 0.0

    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in records)

    is_cache = any(r.get("cache_hit") is not None for r in records if r)

    type_groups: dict[str, list[dict]] = {}
    for r in records:
        if r:
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
        f"| Mean latency | {l_stats['mean']:.2f} s |",
        f"| p50 latency | {l_stats['p50']:.2f} s |",
        f"| p95 latency | {l_stats['p95']:.2f} s |",
        f"| p99 latency | {l_stats['p99']:.2f} s |",
        f"| Min latency | {l_stats['min']:.2f} s |",
        f"| Max latency | {l_stats['max']:.2f} s |",
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
            if r:
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
        g_cite = sum(1 for r in group if r.get("has_citation")) / g_n if g_n else 0.0
        lines.append(f"| {qtype} | {g_n} | {g_score:.3f} | {g_cite:.1%} |")

    diff_groups: dict[str, list[dict]] = {}
    for r in records:
        if r:
            diff_groups.setdefault(r.get("difficulty", "?"), []).append(r)

    lines += [
        "",
        "## Per Difficulty",
        "",
        "| Difficulty | Count | Mean Score | p50 Latency |",
        "|------------|-------|------------|-------------|",
    ]
    for diff in ["easy", "medium", "hard"]:
        group = diff_groups.get(diff, [])
        g_n = len(group)
        if g_n:
            g_score = sum(r["judge_score"] for r in group) / g_n
            g_lat = latency_stats([r["latency_s"] for r in group])["p50"]
            lines.append(f"| {diff} | {g_n} | {g_score:.3f} | {g_lat:.2f} s |")

    rel_scores = [
        r["retrieval_relevance"] for r in records if r and "retrieval_relevance" in r
    ]
    if rel_scores:
        mean_rel = sum(rel_scores) / len(rel_scores)
        full_rel = sum(1 for s in rel_scores if s == 2)
        partial_rel = sum(1 for s in rel_scores if s == 1)
        no_rel = sum(1 for s in rel_scores if s == 0)
        lines += [
            "",
            "## Retrieval Relevance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean relevance score (0-2) | {mean_rel:.2f} |",
            f"| Fully relevant (2) | {full_rel} ({full_rel / len(rel_scores):.1%}) |",
            f"| Partially relevant (1) | {partial_rel} ({partial_rel / len(rel_scores):.1%}) |",
            f"| Irrelevant (0) | {no_rel} ({no_rel / len(rel_scores):.1%}) |",
        ]

    tier_records = [r for r in records if r and r.get("paraphrase_tier")]
    if tier_records:
        tier_groups: dict[str, list[dict]] = {}
        for r in tier_records:
            tier_groups.setdefault(r["paraphrase_tier"], []).append(r)

        lines += [
            "",
            "## Paraphrase Tier Analysis",
            "",
            "| Tier | Count | Cache Hits | Hit Rate | Mean Score |",
            "|------|-------|------------|----------|------------|",
        ]
        for tier in ["easy", "medium", "hard"]:
            group = tier_groups.get(tier, [])
            g_n = len(group)
            if g_n:
                g_hits = sum(1 for r in group if r.get("cache_hit"))
                g_hit_rate = g_hits / g_n
                g_score = sum(r["judge_score"] for r in group) / g_n
                lines.append(
                    f"| {tier} | {g_n} | {g_hits} | {g_hit_rate:.1%} | {g_score:.3f} |"
                )

    fp_records = [r for r in records if r and r.get("false_positive")]
    if fp_records:
        lines += [
            "",
            "## False Positive Error Analysis",
            "",
            "| Question | Cached Source | Expected Source | Match Score |",
            "|----------|--------------|-----------------|-------------|",
        ]
        for r in fp_records:
            q_preview = (
                r["question"][:80] + "..." if len(r["question"]) > 80 else r["question"]
            )
            lines.append(
                f"| {q_preview} | {r.get('cached_question_id', '?')} "
                f"| {r.get('question_id', '?')} | {r.get('cache_match_score', '?')} |"
            )

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


# ---------------------------------------------------------------------------
# Multi-run sweep
# ---------------------------------------------------------------------------


def run_sweep(
    config_path: str | Path,
    *,
    seeds: list[int] | None = None,
    thresholds: list[float] | None = None,
    workload_mixes: list[dict] | None = None,
) -> list[dict]:
    import copy
    import statistics

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    llm_model = os.getenv("LLM_MODEL")
    if llm_model:
        base_config["model"] = llm_model

    judge_model_env = os.getenv("JUDGE_MODEL")
    if judge_model_env:
        base_config["judge_model"] = judge_model_env

    if seeds is None:
        seeds = [42]
    if thresholds is None:
        thresholds = [base_config.get("cache_threshold", 0.92)]
    if workload_mixes is None:
        wl = base_config.get("workload", {})
        workload_mixes = [
            {
                "repeated_query_rate": wl.get("repeated_query_rate", 0.4),
                "paraphrase_rate": wl.get("paraphrase_rate", 0.2),
                "new_query_rate": wl.get("new_query_rate", 0.4),
            }
        ]

    dataset_path = Path(base_config["dataset"])
    base_questions = load_dataset(dataset_path)
    pricing = load_pricing()

    all_summaries: list[dict] = []

    for threshold in thresholds:
        for mix in workload_mixes:
            run_summaries: list[dict] = []
            for seed in seeds:
                config = copy.deepcopy(base_config)
                config["cache_threshold"] = threshold
                wl_config = config.setdefault("workload", {})
                wl_config["seed"] = seed
                wl_config.update(mix)

                run_label = (
                    f"t={threshold}_r={mix['repeated_query_rate']}"
                    f"_p={mix['paraphrase_rate']}_s={seed}"
                )

                # Sweep always creates fresh files (no resume)
                jsonl_path, report_path, _, _ = _prepare_output(f"sweep_{run_label}")

                architecture = config["architecture"]
                if architecture == "semantic_cache_rag":
                    from cag_lab.benchmark.workload import generate_workload

                    near_dupe_pairs = config.get("near_duplicate_pairs")
                    workload = generate_workload(
                        base_questions,
                        repeated_query_rate=mix["repeated_query_rate"],
                        paraphrase_rate=mix["paraphrase_rate"],
                        new_query_rate=mix["new_query_rate"],
                        seed=seed,
                        near_duplicate_pairs=near_dupe_pairs,
                        paraphrase_mode=wl_config.get("paraphrase_mode", "template"),
                        paraphrase_model=wl_config.get(
                            "paraphrase_model", "gpt-4o-mini"
                        ),
                    )
                    records = _run_semantic_cache_rag(
                        config, workload, pricing, jsonl_path, report_path, set(), []
                    )
                elif architecture == "classic_rag":
                    records = _run_classic_rag(
                        config,
                        base_questions,
                        pricing,
                        jsonl_path,
                        report_path,
                        set(),
                        [],
                    )
                else:
                    raise ValueError(f"Sweep not supported for: {architecture}")

                _write_report(report_path, config, records, pricing, jsonl_path.name)

                scores = [r["judge_score"] for r in records if r]
                latencies = [r["latency_s"] for r in records if r]
                costs = [r.get("cost", 0.0) for r in records]
                n = len([r for r in records if r])

                summary = {
                    "run_label": run_label,
                    "threshold": threshold,
                    "mix": mix,
                    "seed": seed,
                    "n": n,
                    "mean_score": sum(scores) / n if n else 0,
                    "mean_latency": sum(latencies) / n if n else 0,
                    "total_cost": sum(costs),
                    "cost_per_1k": (sum(costs) / n * 1000) if n else 0,
                }

                if any(r.get("cache_hit") is not None for r in records if r):
                    summary["cache_hit_rate"] = cache_hit_rate(records)
                    summary["false_positive_rate"] = false_positive_hit_rate(records)
                    summary["cost_saved"] = cost_saved(records)

                run_summaries.append(summary)

            if len(run_summaries) > 1:
                agg = _aggregate_runs(run_summaries)
                agg["threshold"] = threshold
                agg["mix"] = mix
                all_summaries.append(agg)
            else:
                all_summaries.append(run_summaries[0])

    _write_sweep_summary(all_summaries)
    return all_summaries


def _aggregate_runs(summaries: list[dict]) -> dict:
    import statistics

    def _agg(key: str) -> dict:
        values = [s[key] for s in summaries if key in s]
        if not values:
            return {"mean": 0, "std": 0}
        m = statistics.mean(values)
        s = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": round(m, 6), "std": round(s, 6)}

    return {
        "run_label": f"aggregate_{len(summaries)}_runs",
        "n_runs": len(summaries),
        "mean_score": _agg("mean_score"),
        "mean_latency": _agg("mean_latency"),
        "total_cost": _agg("total_cost"),
        "cost_per_1k": _agg("cost_per_1k"),
        "cache_hit_rate": _agg("cache_hit_rate"),
        "false_positive_rate": _agg("false_positive_rate"),
        "cost_saved": _agg("cost_saved"),
    }


def _write_sweep_summary(summaries: list[dict]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"sweep_summary_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    md_path = out_dir / f"sweep_summary_{ts}_report.md"
    lines = [
        "# Sweep Summary",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Runs:** {len(summaries)}",
        "",
    ]

    for s in summaries:
        lines.append(f"## {s['run_label']}")
        lines.append("")
        if "n_runs" in s:
            lines.append(f"Aggregated over {s['n_runs']} seeds")
            lines.append("")
            lines.append("| Metric | Mean | Std Dev |")
            lines.append("|--------|------|---------|")
            for key in [
                "mean_score",
                "mean_latency",
                "cost_per_1k",
                "cache_hit_rate",
                "false_positive_rate",
            ]:
                val = s.get(key, {})
                if isinstance(val, dict):
                    lines.append(
                        f"| {key} | {val.get('mean', '-')} | {val.get('std', '-')} |"
                    )
        else:
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key in [
                "mean_score",
                "mean_latency",
                "cost_per_1k",
                "cache_hit_rate",
                "false_positive_rate",
            ]:
                val = s.get(key, "-")
                if isinstance(val, float):
                    lines.append(f"| {key} | {val:.4f} |")
                else:
                    lines.append(f"| {key} | {val} |")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
