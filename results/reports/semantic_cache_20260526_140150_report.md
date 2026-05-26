# Semantic-Cache RAG — gpt-4o-mini @ top-5 (threshold=0.92)

**Date:** 2026-05-26 14:01 UTC
**Results file:** semantic_cache_20260526_140150.jsonl

## Configuration

| Key | Value |
|-----|-------|
| Architecture | semantic_cache_rag |
| Dataset | data/benchmark_sets/aws_docs_v1.jsonl |
| Index | aws-docs |
| Model | gpt-4o-mini |
| Top-K | 5 |
| Questions | 302 |
| Cache threshold | 0.92 |
| Cache TTL | 3600 s |
| Repeat rate | 0.4 |
| Paraphrase rate | 0.2 |
| New rate | 0.4 |

## Aggregate Quality

| Metric | Value |
|--------|-------|
| Mean judge score | 0.351 |
| Citation rate | 58.6% |
| p50 latency | 0.26 s |
| p95 latency | 6.95 s |
| Total tokens | 320,620 (305,371 prompt + 15,249 completion) |
| Total cost | $0.054955 |
| Cost per 1k questions | $0.181970 |

## Cache Performance

| Metric | Value |
|--------|-------|
| Cache hit rate | 60.6% |
| False-positive hit rate | 3.3% |
| Cost saved | $0.083280 |

### Hit rate by relationship

| Relationship | Count | Hit Rate |
|--------------|-------|----------|
| exact | 120 | 71.7% |
| paraphrase | 60 | 61.7% |
| new | 122 | 49.2% |

## Per Query Type

| Query Type | Count | Mean Score | Citation Rate |
|------------|-------|------------|---------------|
| comparison | 50 | 0.440 | 64.0% |
| factual | 51 | 0.314 | 72.5% |
| multi_hop | 35 | 0.200 | 65.7% |
| procedural | 30 | 0.700 | 80.0% |
| troubleshooting | 136 | 0.294 | 44.9% |
