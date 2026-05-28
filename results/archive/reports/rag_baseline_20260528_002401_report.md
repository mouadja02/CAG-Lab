# RAG Baseline — nvidia/nemotron-nano-9b-v2 @ top-5

**Date:** 2026-05-28 01:00 UTC
**Results file:** rag_baseline_20260528_002401.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | classic_rag |
| Dataset | data/benchmark_sets/experiment_dataset.jsonl |
| Index | aws-docs |
| Model | nvidia/nemotron-nano-9b-v2 |
| Top-K | 5 |
| Questions | 200 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.690 |
| Citation rate | 89.5% |
| Mean latency | 9.14 s |
| p50 latency | 5.05 s |
| p95 latency | 20.22 s |
| p99 latency | 127.01 s |
| Min latency | 2.74 s |
| Max latency | 127.93 s |
| Total tokens | 670,092 (468,659 prompt + 201,433 completion) |
| Total cost | $0.050976 |
| Cost per 1k questions | $0.254878 |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 19 | 0.789 | 89.5% |
| factual | 81 | 0.642 | 88.9% |
| multi_hop | 15 | 0.933 | 86.7% |
| procedural | 65 | 0.646 | 90.8% |
| troubleshooting | 20 | 0.750 | 90.0% |

## Per Difficulty

| Difficulty | Count | Mean Score | p50 Latency |
|------------|-------|------------|-------------|
| easy | 54 | 0.648 | 4.53 s |
| medium | 133 | 0.684 | 5.22 s |
| hard | 13 | 0.923 | 5.41 s |

## Retrieval Relevance

| Metric | Value |
|--------|-------|
| Mean relevance score (0-2) | 1.83 |
| Fully relevant (2) | 181 (90.5% if rel_scores else '-') |
| Partially relevant (1) | 5 (2.5% if rel_scores else '-') |
| Irrelevant (0) | 14 (7.0% if rel_scores else '-') |
