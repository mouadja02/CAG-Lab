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

## Why Qdrant? Why Migrate from Pinecone?

The original pipeline relied on **Pinecone** (cloud-hosted vector DB) for retrieval
and **Redis Stack** for the semantic cache. This meant every experiment required:

- A stable internet connection
- A Pinecone subscription
- Network latency between the local machine and Pinecone's servers

**Research should be reproducible offline.** A benchmark that depends on a cloud
service isn't fully reproducible — someone else cloning the repo can't run the exact
same experiments without their own Pinecone credentials and an index populated with
the same vectors.

We migrated the retrieval layer from Pinecone to **Qdrant**, an open-source vector
database that runs locally via Docker, for three reasons:

1. **Zero-dependency testing** — Both retrieval (Qdrant) and caching (Redis) now run
   locally. No cloud services, no API quotas, no network flakiness.
2. **Reproducible science** — Anyone can clone the repo, run `docker compose up -d`,
   `python scripts/migrate_pinecone_to_qdrant.py --force`, and get identical results.
3. **Same capabilities** — Qdrant supports the same cosine-distance vector search with
   512-dimensional embeddings as Pinecone, matching our `text-embedding-3-small`
   configuration exactly.

The original Pinecone path remains available — set `VECTOR_DB=pinecone` (the default)
to use the cloud index. The migration script preserves every vector and its metadata,
allowing seamless switching between backends for A/B comparison of retrieval latency.

## What We Compare

| Architecture | Description |
|---|---|
| **Classic RAG** | Embed query → retrieve top-K from vector DB → generate with LLM |
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
# source .venv/bin/activate  # macOS/Linux
pip install -e .

# Start local services (Redis for cache, Qdrant for retrieval)
docker compose up -d

# Configure — copy .env.example to .env and fill in your API keys
cp .env.example .env

# --- Option A: Internet-independent (recommended for reproducibility) ---
# Set Qdrant as vector backend and migrate the Pinecone index locally
$env:VECTOR_DB = "qdrant"   # Windows; use export on macOS/Linux
python scripts/migrate_pinecone_to_qdrant.py --index aws-docs --force

# --- Option B: Use Pinecone cloud (requires PINECONE_API_KEY) ---
# $env:VECTOR_DB = "pinecone"  # this is the default

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
│   ├── generate_comparison_report.py
│   └── migrate_pinecone_to_qdrant.py  # Pinecone → Qdrant migration
├── src/
│   └── cag_lab/
│       ├── benchmark/      # Dataset, metrics, runner, workload
│       ├── cache/          # Redis vector search semantic cache
│       ├── generation/     # LLM client + answer generator
│       └── retrieval/      # Qdrant & Pinecone retrievers (VECTOR_DB env switch)
├── docker-compose.yml      # Redis Stack + Qdrant
└── pyproject.toml
```

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | OpenAI `text-embedding-3-small` (512d) |
| Vector DB | Qdrant (default, local) / Pinecone (`aws-docs` index, cloud) |
| Cache | Redis Stack (HNSW, cosine distance) |
| LLM | GPT-4o-mini via OpenAI client (OpenRouter-compatible) |
| Judge | GPT-4o-mini |
| CLI | Typer |
| Dashboard | Vanilla HTML/CSS + Chart.js |

## Pinecone → Qdrant Migration

### Motivation

Before this migration, every experiment required an active Pinecone cloud subscription
and a stable internet connection. This violated a core principle of reproducible
research: **anyone cloning the repo should be able to reproduce the exact same results**
without signing up for cloud services.

### What We Migrated

| From | To |
|---|---|
| **Pinecone** (cloud, 89,221 vectors) | **Qdrant** (local, Docker) |
| 512-dim `aws-docs` index | 512-dim `aws-docs` collection |
| Backslash path IDs (`documents\AWS-...`) | Deterministic UUIDs (`uuid5` from original ID) |
| Metadata: `content`, `filePath`, `chunkIndex`, etc. | Exact copy (plus `_pinecone_id` for traceability) |

### How It Works

```
┌─────────────┐     list() / fetch()      ┌───────────┐     upsert()      ┌───────────┐
│  Pinecone    │ ────────────────────────→ │  Migration │ ───────────────→ │  Qdrant   │
│  cloud index │ ←──── vectors + metadata  │  script    │ ←──── UUIDs     │  (local)  │
└─────────────┘                           └───────────┘                  └───────────┘
```

1. **List** — Pinecone's `list(prefix="")` paginates through all 89,221 vector IDs
   in batches
2. **Fetch** — Each batch of IDs is fetched with `index.fetch(ids=...)` to retrieve
   the full 512-dim embedding vectors and metadata payloads
3. **Transform IDs** — Pinecone uses Windows-style file paths as IDs (e.g.
   `documents\AWS-Kinesis\...`), which Qdrant rejects. We generate deterministic
   UUIDs via `uuid.uuid5(uuid.NAMESPACE_URL, original_id)` and store the original
   ID as `_pinecone_id` in the payload
4. **Upsert** — All 89,221 points are uploaded to a local Qdrant collection with
   cosine-distance vector index, matching the original Pinecone configuration

### Switching Between Backends

The retriever is backend-agnostic. Set the `VECTOR_DB` environment variable:

```bash
# Use local Qdrant (default for offline experiments)
export VECTOR_DB=qdrant

# Use Pinecone cloud (original backend)
export VECTOR_DB=pinecone  # or leave unset (this is the default)
```

Both backends expose the identical `Chunk` and `Retriever` interface — the only
difference is which Docker container the vectors live in. This also lets us measure
retrieval latency differences between local and cloud vector search in future
experiments.

## Roadmap

This is an active playground — here's what's coming:

- [x] **Local vector DB** — Replaced Pinecone cloud with Qdrant (Docker) for
  internet-independent benchmarking. VECTOR_DB env switch for A/B comparison.
- [x] **Direct OpenAI client** — Replaced LiteLLM with native `openai.OpenAI` pointing
  at OpenRouter, eliminating noisy provider-discovery warnings.
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
