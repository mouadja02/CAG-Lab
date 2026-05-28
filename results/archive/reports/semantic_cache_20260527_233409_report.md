# Semantic-Cache RAG — nvidia/nemotron-nano-9b-v2 @ top-5 (threshold=0.92)

**Date:** 2026-05-27 23:34 UTC
**Results file:** semantic_cache_20260527_233409.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | semantic_cache_rag |
| Dataset | data/benchmark_sets/aws_docs_v3.jsonl |
| Index | aws-docs |
| Model | nvidia/nemotron-nano-9b-v2 |
| Top-K | 5 |
| Questions | 2500 |
| Cache threshold | 0.92 |
| Cache TTL | 3600 s |
| Repeat rate | 0.4 |
| Paraphrase rate | 0.2 |
| New rate | 0.4 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.741 |
| Citation rate | 88.9% |
| p50 latency | 4.19 s |
| p95 latency | 10.55 s |
| Total tokens | 4,777,142 (3,639,272 prompt + 1,137,870 completion) |
| Total cost | $0.000000 |
| Cost per 1k questions | $0.000000 |

## Cache Performance

| Metric | Value |
|--------|-------|
| Cache hit rate | 30.8% |
| False-positive hit rate | 0.6% |
| Cost saved | $0.539700 |

### Hit rate by relationship

| Relationship | Count | Hit Rate |
|--------------|-------|----------|
| exact | 1000 | 34.3% |
| paraphrase | 500 | 37.6% |
| new | 1000 | 24.0% |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 57 | 0.754 | 87.7% |
| factual | 1305 | 0.726 | 86.3% |
| multi_hop | 50 | 0.780 | 94.0% |
| procedural | 950 | 0.755 | 91.9% |
| troubleshooting | 138 | 0.761 | 92.0% |

## Per Difficulty

| Difficulty | Count | Mean Score | p50 Latency |
|------------|-------|------------|-------------|
| easy | 1266 | 0.749 | 3.95 s |
| medium | 1189 | 0.734 | 4.50 s |
| hard | 45 | 0.689 | 5.32 s |

## Paraphrase Tier Analysis

| Tier | Count | Cache Hits | Hit Rate | Mean Score |
|------|-------|------------|----------|------------|
| easy | 500 | 188 | 37.6% | 0.742 |

## False Positive Error Analysis

| Question | Cached Source | Expected Source | Match Score |
|----------|--------------|-----------------|-------------|
| When launching Spot Instances in a Nondefault VPC, what must you provide to ensu... | aws-862 | aws-863 | 0.9526 |
| What should contributors do before submitting any pull requests or issues? | aws-520 | aws-519 | 0.929 |
| Elaborate on: When launching Spot Instances in a Nondefault VPC, what must you p... | aws-862 | aws-863 | 0.9223 |
| What is the default name of the task instance that can be modified? | aws-123 | aws-122 | 0.9567 |
| If a fleet targets 16 On-Demand instances and only 15 unused Capacity Reservatio... | aws-272 | aws-273 | 0.922 |
