# RAG Baseline — nvidia/nemotron-nano-9b-v2 @ top-5

**Date:** 2026-05-28 07:47 UTC
**Results file:** rag_baseline_20260528_071701.jsonl

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
| Mean judge score | 0.650 |
| Citation rate | 87.5% |
| Mean latency | 7.50 s |
| p50 latency | 5.06 s |
| p95 latency | 14.66 s |
| p99 latency | 27.72 s |
| Min latency | 2.80 s |
| Max latency | 129.62 s |
| Total tokens | 627,419 (468,659 prompt + 158,760 completion) |
| Total cost | $0.044148 |
| Cost per 1k questions | $0.220740 |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 19 | 0.789 | 89.5% |
| factual | 81 | 0.617 | 85.2% |
| multi_hop | 15 | 0.867 | 93.3% |
| procedural | 65 | 0.646 | 87.7% |
| troubleshooting | 20 | 0.500 | 90.0% |

## Per Difficulty

| Difficulty | Count | Mean Score | p50 Latency |
|------------|-------|------------|-------------|
| easy | 54 | 0.667 | 4.71 s |
| medium | 133 | 0.624 | 5.13 s |
| hard | 13 | 0.846 | 5.13 s |

## Retrieval Relevance

| Metric | Value |
|--------|-------|
| Mean relevance score (0-2) | 1.83 |
| Fully relevant (2) | 179 (89.5%) |
| Partially relevant (1) | 9 (4.5%) |
| Irrelevant (0) | 12 (6.0%) |
