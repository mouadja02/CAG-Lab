# Semantic-Cache RAG — nvidia/nemotron-nano-9b-v2 @ top-5 (threshold=0.96)

**Date:** 2026-05-28 08:34 UTC
**Results file:** semantic_cache_20260528_074751.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | semantic_cache_rag |
| Dataset | data/benchmark_sets/experiment_dataset.jsonl |
| Index | aws-docs |
| Model | nvidia/nemotron-nano-9b-v2 |
| Top-K | 5 |
| Questions | 500 |
| Cache threshold | 0.96 |
| Cache TTL | 3600 s |
| Repeat rate | 0.4 |
| Paraphrase rate | 0.2 |
| New rate | 0.4 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.686 |
| Citation rate | 90.6% |
| Mean latency | 4.44 s |
| p50 latency | 0.56 s |
| p95 latency | 10.42 s |
| p99 latency | 22.22 s |
| Min latency | 0.42 s |
| Max latency | 172.72 s |
| Total tokens | 716,556 (520,925 prompt + 195,631 completion) |
| Total cost | $0.052138 |
| Cost per 1k questions | $0.104276 |

## Cache Performance

| Metric | Value |
|--------|-------|
| Cache hit rate | 54.6% |
| False-positive hit rate | 0.0% |
| Cost saved | $0.056689 |

### Hit rate by relationship

| Relationship | Count | Hit Rate |
|--------------|-------|----------|
| exact | 200 | 100.0% |
| paraphrase | 100 | 73.0% |
| new | 200 | 0.0% |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 42 | 0.738 | 83.3% |
| factual | 206 | 0.631 | 89.8% |
| multi_hop | 36 | 0.833 | 94.4% |
| procedural | 166 | 0.717 | 94.0% |
| troubleshooting | 50 | 0.660 | 86.0% |

## Per Difficulty

| Difficulty | Count | Mean Score | p50 Latency |
|------------|-------|------------|-------------|
| easy | 134 | 0.701 | 3.55 s |
| medium | 335 | 0.663 | 0.55 s |
| hard | 31 | 0.871 | 0.56 s |

## Paraphrase Tier Analysis

| Tier | Count | Cache Hits | Hit Rate | Mean Score |
|------|-------|------------|----------|------------|
| easy | 100 | 73 | 73.0% | 0.730 |
