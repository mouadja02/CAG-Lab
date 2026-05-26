# CAG-Lab

A **personal learning playground** for benchmarking Retrieval-Augmented Generation (RAG)
against Cache-Augmented Generation (CAG) on real-world AWS documentation queries.
I built this to satisfy my own curiosity — but the results turned out to be worth sharing.

> **Live dashboard:** [mouadja02.github.io/cag-lab](https://mouadja02.github.io/cag-lab)

## Why This Exists

I kept seeing signals converge — a [YouTube video](https://youtu.be/QA3h4H5jqJw), a
[couple](https://x.com/akshay_pachaar/status/2056714042455343160)
[of](https://x.com/techNmak/status/2006727285223886937)
posts, and two
[papers](https://arxiv.org/abs/2412.15605)
([MeanCache](https://arxiv.org/abs/2403.02694), [Don't Do RAG](https://arxiv.org/abs/2412.15605))
— all pointing at the same idea: **put a semantic cache between retrieval and
generation**. As a system architect, this looks exactly like the pattern we use every
day — Redis or Memcached between an app server and PostgreSQL. Can't we do the same
for LLM pipelines?

This repo is my sandbox to test that hypothesis rigorously: same dataset, same model,
side-by-side comparison, every metric tracked.

## Why GPT-4o-mini?

We deliberately use **gpt-4o-mini**, the smallest and cheapest OpenAI model. The point
is *not* to see how well an LLM can answer AWS questions from its training data.
A frontier model like GPT-4o or Claude would ace many of these questions from general
knowledge alone, making retrieval irrelevant.

We want a model that **must rely on the retrieved chunks** to answer correctly.
This way, the benchmark actually measures *retrieval quality* — not model memorization.

## What We Compare

| Architecture | Description |
|---|---|
| **Classic RAG** | Embed query → retrieve top-K from Pinecone → generate with LLM |
| **Semantic Cache (CAG)** | Embed query → Redis vector search → HIT: return cached / MISS: run RAG + store |

Coming next: long-context CAG (preload all docs into the context window + cache KV state).

## What We Measure

- **Answer quality** — LLM judge (GPT-4o-mini) scores correctness 0/1 per question
- **Latency** — p50/p95 wall-clock time per question
- **Cost** — token-based pricing from `configs/models/pricing.yaml`
- **Citation rate** — does the answer cite its sources?
- **Cache hit rate** — fraction served from Redis (CAG only)
- **False-positive rate** — cache hits where the source question was actually different
- **Cost saved** — generation cost avoided by caching (CAG only)

## Latest Results

| Metric | RAG Baseline | Semantic Cache | Improvement |
|---|---|---|---|
| Mean judge score | 0.333 | 0.351 | +5% |
| p50 latency | 4.01 s | 0.26 s | **−93%** |
| Cost per 1k Qs | $0.455 | $0.182 | **−60%** |
| Cache hit rate | — | 60.6% | — |
| False-positive rate | — | 3.3% | — |

> Full interactive charts and per-query-type breakdown on the
> [dashboard](https://mouadja02.github.io/cag-lab/report.html).

## Quick Start

```bash
git clone https://github.com/mouadja02/cag-lab.git
cd cag-lab

# Install
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -e .

# Start Redis (required for semantic cache experiments)
docker compose up -d

# Configure — copy .env.example to .env and fill in your API keys
cp .env.example .env

# Run the RAG baseline (120 questions)
cag-lab run --config configs/experiments/rag_baseline.yaml

# Run with semantic cache (302 workload items)
cag-lab run --config configs/experiments/semantic_cache.yaml

# Generate the comparison report
python scripts/generate_comparison_report.py --auto
```

## Project Structure

```
cag-lab/
├── configs/
│   ├── experiments/        # YAML experiment definitions
│   └── models/             # Pricing table (editable)
├── data/
│   └── benchmark_sets/     # JSONL Q&A datasets
├── docs/                   # GitHub Pages dashboard
│   ├── index.html          # Landing page
│   ├── rag.html            # RAG deep-dive + SVG architecture
│   ├── cag.html            # CAG deep-dive + research citations
│   ├── report.html         # Auto-generated comparison (charts)
│   └── report.md           # Downloadable markdown report
├── results/
│   ├── jsonl/              # Raw experiment results (committed)
│   └── reports/            # Per-experiment markdown reports
├── scripts/
│   └── generate_comparison_report.py
├── src/
│   └── cag_lab/
│       ├── benchmark/      # Dataset, metrics, runner, workload
│       ├── cache/          # Redis vector search semantic cache
│       ├── generation/     # LLM client + answer generator
│       └── retrieval/      # Pinecone retrieval
├── docker-compose.yml      # Redis Stack (with vector search)
└── pyproject.toml
```

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | OpenAI `text-embedding-3-small` (512d) |
| Vector DB | Pinecone (`aws-docs` index) |
| Cache | Redis Stack (HNSW, cosine distance) |
| LLM | GPT-4o-mini via LiteLLM |
| Judge | GPT-4o-mini |
| CLI | Typer |
| Dashboard | Vanilla HTML/CSS + Chart.js |

## Roadmap

This is an active playground — here's what's coming:

- [ ] **Local SLMs** — Test with Ollama/LM Studio models (Llama, Mistral, Phi) to
  eliminate API costs and measure on-device performance
- [ ] **Long-context CAG** — Preload the entire knowledge base into the context window,
  cache KV state, skip retrieval entirely (per Don't Do RAG paper)
- [ ] **Bigger benchmark** — Expand the dataset beyond 120 AWS questions; add Azure,
  GCP, and Kubernetes documentation domains
- [ ] **More query types** — Add conversational multi-turn, contradictory questions,
  adversarial queries
- [ ] **Different embedding models** — Compare OpenAI vs Cohere vs open-source
  embeddings on retrieval quality
- [ ] **Multi-model comparison** — GPT-4o, Claude, Gemini side-by-side with the same
  retrieval pipeline
- [ ] **Cost-tracking dashboard** — Real-time cost monitoring per experiment run
- [ ] **Streaming mode** — Measure time-to-first-token alongside full answer latency
- [ ] **Eviction policies** — LRU, LFU, and score-based cache eviction strategies
- [ ] **Hybrid search** — Dense + sparse (BM25) retrieval and caching

## How It Works (CI/CD)

Every push to `main` that touches `results/jsonl/`, `scripts/`, or `docs/` triggers a
GitHub Action:

1. Runs `scripts/generate_comparison_report.py --auto` to discover the latest two
   experiment JSONL files
2. Generates `docs/report.html` (interactive charts) and `docs/report.md`
3. Deploys the `docs/` directory to GitHub Pages at
   [mouadja02.github.io/cag-lab](https://mouadja02.github.io/cag-lab)

## License

MIT — see [LICENSE](LICENSE). The benchmark dataset is derived from public AWS
documentation.
