# Semantic-Cache RAG — nvidia/nemotron-nano-9b-v2 @ top-5 (threshold=0.92)

**Date:** 2026-05-28 01:42 UTC
**Results file:** semantic_cache_20260528_010006.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | semantic_cache_rag |
| Dataset | data/benchmark_sets/experiment_dataset.jsonl |
| Index | aws-docs |
| Model | nvidia/nemotron-nano-9b-v2 |
| Top-K | 5 |
| Questions | 500 |
| Cache threshold | 0.92 |
| Cache TTL | 3600 s |
| Repeat rate | 0.4 |
| Paraphrase rate | 0.2 |
| New rate | 0.4 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.680 |
| Citation rate | 88.0% |
| Mean latency | 3.96 s |
| p50 latency | 0.48 s |
| p95 latency | 10.05 s |
| p99 latency | 22.37 s |
| Min latency | 0.41 s |
| Max latency | 128.59 s |
| Total tokens | 642,539 (462,793 prompt + 179,746 completion) |
| Total cost | $0.047271 |
| Cost per 1k questions | $0.094542 |

## Cache Performance

| Metric | Value |
|--------|-------|
| Cache hit rate | 60.0% |
| False-positive hit rate | 0.7% |
| Cost saved | $0.063576 |

### Hit rate by relationship

| Relationship | Count | Hit Rate |
|--------------|-------|----------|
| exact | 200 | 100.0% |
| paraphrase | 100 | 99.0% |
| new | 200 | 0.5% |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 42 | 0.738 | 83.3% |
| factual | 206 | 0.636 | 87.4% |
| multi_hop | 36 | 0.861 | 91.7% |
| procedural | 166 | 0.711 | 88.6% |
| troubleshooting | 50 | 0.580 | 90.0% |

## Per Difficulty

| Difficulty | Count | Mean Score | p50 Latency |
|------------|-------|------------|-------------|
| easy | 134 | 0.672 | 0.49 s |
| medium | 335 | 0.663 | 0.48 s |
| hard | 31 | 0.903 | 0.48 s |

## Paraphrase Tier Analysis

| Tier | Count | Cache Hits | Hit Rate | Mean Score |
|------|-------|------------|----------|------------|
| easy | 100 | 99 | 99.0% | 0.770 |

## False Positive Error Analysis

| Question | Cached Source | Expected Source | Match Score |
|----------|--------------|-----------------|-------------|
| If a fleet targets 16 On-Demand instances and only 15 unused Capacity Reservatio... | aws-192 | aws-107 | 0.922 |
| If a fleet targets 16 On-Demand instances and only 15 unused Capacity Reservatio... | aws-192 | aws-107 | 0.922 |
