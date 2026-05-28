# CAG-Lab Benchmark Comparison: RAG Baseline vs Semantic Cache

*Generated 2026-05-28 09:50 UTC*

## Summary

| Metric | RAG Baseline | Semantic Cache | Change |
|--------|----------|----------|--------|
| Questions | 200 | 500 | +150% |
| Mean Judge Score | 0.650 | 0.686 | +6% |
| Citation Rate | 87.5% | 90.6% | +4% |
| p50 Latency | 5.06 s | 0.56 s | -89% |
| p95 Latency | 14.66 s | 10.42 s | -29% |
| p99 Latency | 129.62 s | 172.72 s | +33% |
| Mean Latency | 7.50 s | 4.44 s | -41% |
| Total Tokens | 627,419 | 716,556 | +14% |
| Total Cost | 0.044148 $ | 0.052138 $ | +18% |
| Cost per 1k Q | 0.220740 $ | 0.104276 $ | -53% |

## Cache Performance

| Metric | Value |
|--------|-------|
| Hit rate | 54.6% (273/500) |
| False-positive rate | 0.0% (0 of 273 hits) |
| Cost saved | $0.056689 |
| Hit p50 latency | 0.46 s |
| Miss p50 latency | 5.63 s |

### Hit Rate by Relationship

| Relationship | Count | Hits | Hit Rate |
|--------------|-------|------|----------|
| exact | 200 | 200 | 100.0% |
| new | 200 | 0 | 0.0% |
| paraphrase | 100 | 73 | 73.0% |

## Per Query Type

| Query Type | RAG Baseline Score | Semantic Cache Score | Δ |
|------------|--------------|--------------|---|
| comparison | 0.789 | 0.738 | -0.051 |
| factual | 0.617 | 0.631 | +0.014 |
| multi_hop | 0.867 | 0.833 | -0.033 |
| procedural | 0.646 | 0.717 | +0.071 |
| troubleshooting | 0.500 | 0.660 | +0.160 |

