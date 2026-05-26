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

from cag_lab.benchmark.dataset import load_dataset
from cag_lab.benchmark.metrics import (
    citation_present,
    compute_question_cost,
    latency_stats,
    llm_judge_correctness,
    load_pricing,
)


def _run_classic_rag(config: dict, questions, pricing: dict):
    from cag_lab.generation.answer_generator import generate_answer
    from cag_lab.retrieval.pinecone_retriever import Retriever

    model = config["model"]
    index_name = config["index"]
    top_k = config["top_k"]

    retriever = Retriever(index_name=index_name, top_k=top_k)

    records = []
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


def run_experiment(config_path: str | Path) -> None:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config["dataset"])
    questions = load_dataset(dataset_path)

    pricing = load_pricing()

    architecture = config["architecture"]
    if architecture == "classic_rag":
        records = _run_classic_rag(config, questions, pricing)
    elif architecture in ("semantic_cache", "long_context"):
        raise NotImplementedError(f"Architecture '{architecture}' is not yet implemented.")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    _write_results(config_path.stem, config, questions, records, pricing)


def _write_results(experiment_name, config, questions, records, pricing):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    jsonl_path = results_dir / f"{experiment_name}_{ts}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    _write_report(results_dir / f"{experiment_name}_{ts}_report.md", config, questions, records, pricing, jsonl_path.name)


def _write_report(report_path, config, questions, records, pricing, jsonl_name):
    model = config["model"]
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
        f"| Architecture | {config['architecture']} |",
        f"| Dataset | {config['dataset']} |",
        f"| Index | {config['index']} |",
        f"| Model | {config['model']} |",
        f"| Top-K | {config['top_k']} |",
        f"| Questions | {n} |",
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
        "",
        "## Per Query Type",
        "",
        "| Query Type | Count | Mean Score | Citation Rate |",
        "|------------|-------|------------|---------------|",
    ]

    for qtype in sorted(type_groups):
        group = type_groups[qtype]
        g_n = len(group)
        g_score = sum(r["judge_score"] for r in group) / g_n
        g_cite = sum(1 for r in group if r["has_citation"]) / g_n
        lines.append(f"| {qtype} | {g_n} | {g_score:.3f} | {g_cite:.1%} |")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def config_path_to_title(config: dict) -> str:
    return f"RAG Baseline — {config['model']} @ top-{config['top_k']}"
