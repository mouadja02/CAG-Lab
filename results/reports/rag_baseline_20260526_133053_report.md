# RAG Baseline — gpt-4o-mini @ top-5

**Date:** 2026-05-26 13:30 UTC
**Results file:** rag_baseline_20260526_133053.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | classic_rag |
| Dataset | data/benchmark_sets/aws_docs_v1.jsonl |
| Index | aws-docs |
| Model | gpt-4o-mini |
| Top-K | 5 |
| Questions | 120 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.333 |
| Citation rate | 56.7% |
| p50 latency | 4.01 s |
| p95 latency | 8.98 s |
| Total tokens | 322,518 (308,692 prompt + 13,826 completion) |
| Total cost | $0.054599 |
| Cost per 1k questions | $0.454995 |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 21 | 0.429 | 52.4% |
| factual | 21 | 0.286 | 76.2% |
| multi_hop | 18 | 0.278 | 61.1% |
| procedural | 10 | 0.700 | 80.0% |
| troubleshooting | 50 | 0.260 | 44.0% |
