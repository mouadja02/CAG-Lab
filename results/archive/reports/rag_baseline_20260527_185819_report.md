# RAG Baseline — nvidia/nemotron-nano-9b-v2 @ top-5

**Date:** 2026-05-27 18:58 UTC
**Results file:** rag_baseline_20260527_185819.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | classic_rag |
| Dataset | data/benchmark_sets/aws_docs_v3.jsonl |
| Index | aws-docs |
| Model | nvidia/nemotron-nano-9b-v2 |
| Top-K | 5 |
| Questions | 1000 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.748 |
| Citation rate | 89.5% |
| p50 latency | 4.44 s |
| p95 latency | 13.89 s |
| Total tokens | 2,789,069 (2,076,195 prompt + 712,874 completion) |
| Total cost | $0.000000 |
| Cost per 1k questions | $0.000000 |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 23 | 0.739 | 91.3% |
| factual | 524 | 0.744 | 86.5% |
| multi_hop | 23 | 0.826 | 91.3% |
| procedural | 378 | 0.741 | 92.1% |
| troubleshooting | 52 | 0.808 | 100.0% |

## Per Difficulty

| Difficulty | Count | Mean Score | p50 Latency |
|------------|-------|------------|-------------|
| easy | 503 | 0.767 | 4.07 s |
| medium | 476 | 0.727 | 4.68 s |
| hard | 21 | 0.762 | 5.41 s |

## Retrieval Relevance

| Metric | Value |
|--------|-------|
| Mean relevance score (0-2) | 1.79 |
| Fully relevant (2) | 884 (88.4%) |
| Partially relevant (1) | 23 (2.3%) |
| Irrelevant (0) | 93 (9.3%) |
