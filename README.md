# CAG-Lab

A **research playground** benchmarking Retrieval-Augmented Generation (RAG)
against Cache-Augmented Generation (CAG) on real AWS documentation queries.
Built from curiosity — turned into a shareable proof-of-concept.

> **Live dashboard:** [mouadja02.github.io/cag-lab](https://mouadja02.github.io/cag-lab)

## Why This Exists

I kept seeing signals converge — a [YouTube video](https://youtu.be/QA3h4H5jqJw),
[posts](https://x.com/techNmak/status/2006727285223886937), and two
[papers](https://arxiv.org/abs/2412.15605)
([MeanCache](https://arxiv.org/abs/2403.02694), [Don't Do RAG](https://arxiv.org/abs/2412.15605))
— all pointing at the same idea: **put a semantic cache between retrieval and
generation**. As a system architect, this looks exactly like the pattern we use
every day — Redis or Memcached between an app server and PostgreSQL. Can't we
do the same for LLM pipelines?

This repo is my sandbox to test that hypothesis rigorously: same dataset, same
model, side-by-side comparison, every metric tracked.

## The Journey

### Phase 1: Baseline (Pinecone + LiteLLM + GPT-4o-mini)

Started with a cloud-only pipeline: Pinecone for vector search, LiteLLM for
model routing, GPT-4o-mini for generation. 120 hand-written AWS questions. This
established the benchmark framework — YAML configs, run_experiment runner, JSONL
outputs, markdown reports, and a GitHub Pages dashboard.

### Phase 2: Internet-Independent Research

Every experiment required a Pinecone subscription and stable internet. This broke
reproducibility — nobody else could clone and run. **Migrated 89,221 vectors from
Pinecone to Qdrant** running locally in Docker. Wrote a migration script that
preserves every vector, handles Pinecone's backslash-path IDs, and supports
`VECTOR_DB=qdrant` / `VECTOR_DB=pinecone` switching for A/B comparison.

### Phase 3: Multi-Provider LLM Support

Replaced LiteLLM's noisy provider-discovery with a direct `openai.OpenAI` client.
Added env-var-based provider switching (`LLM_API_BASE`, `LLM_API_KEY`,
`EMBED_API_BASE`) supporting OpenRouter, OpenAI direct, Anthropic, LMStudio,
Ollama, vLLM — any OpenAI-compatible endpoint. Added `JUDGE_API_KEY` for
separate judge-model routing.

### Phase 4: Dataset Coherence

The original 120 questions were hand-written against a different knowledge base.
The local model answered "I cannot answer" on most questions because retrieved
chunks didn't match. **Rebuilt the dataset from the vector DB itself**: sampled
random vectors from Qdrant, extracted content, and used an LLM judge to generate
1,000 benchmark questions directly from the indexed documents. Filtered to 200
clean, substantive questions with balanced query types and difficulty levels.

### Phase 5: Dataset Sharing

Local Qdrant solved the internet-dependency problem but introduced a new one:
the 89,221 vectors still required a Pinecone subscription to populate. **Exported
the full vector collection to Parquet and published it on
[HuggingFace Datasets](https://huggingface.co/datasets/mouadja/aws-docs)**.
Anyone can now `python scripts/setup_vectordb.py` to download and restore the
collection — no Pinecone, no API keys, just Docker + one command.

## What We Compare

| Architecture | Description |
|---|---|
| **Classic RAG** | Embed query → retrieve top-K from vector DB → generate with LLM |
| **Semantic Cache (CAG)** | Embed query → Redis vector search → HIT: return cached / MISS: run RAG + store |

## What We Measure

- **Answer quality** — LLM judge (GPT-4o-mini) scores correctness 0/1 per question
- **Latency** — p25/p50/p75/p95/p99/min/max/mean wall-clock per question
- **Cost** — token-based pricing from `configs/models/pricing.yaml`, with missing-model warnings
- **Citation rate** — does the answer cite its sources?
- **Cache hit rate** — fraction served from Redis (CAG only), by relationship
- **False-positive rate** — cache hits where the source question was different
- **Cost saved** — generation cost avoided by caching (CAG only)
- **Retrieval relevance** — LLM judge scores whether retrieved chunks contain the answer

## Latest Results

> **200 questions, 500 workload items, threshold 0.96, model nvidia/nemotron-nano-9b-v2**

| Metric | RAG Baseline | Semantic Cache | Improvement |
|---|---|---|---|
| Mean judge score | 0.650 | **0.686** | **+6%** |
| Citation rate | 87.5% | 90.6% | +3pp |
| p50 latency | 5.06 s | **0.56 s** | **−89%** |
| Cost per 1k Qs | $0.221 | **$0.104** | **−53%** |
| Cache hit rate | — | 54.6% | — |
| False-positive rate | — | **0.0%** | — |

> **CAG beats RAG on every dimension**: higher quality (+6%), lower cost (-53%),
> dramatically lower latency (-89%). Zero false positives at threshold 0.96.
> Full interactive charts and per-query-type breakdown on the
> [dashboard](https://mouadja02.github.io/cag-lab/report.html).

## Tech Stack

| Layer | Technology |
|---|---|
| Embedding | OpenAI `text-embedding-3-small` (512d) |
| Vector DB | Qdrant (local, Docker) / Pinecone (cloud) — env-switchable. Vectors on [HuggingFace](https://huggingface.co/datasets/mouadja/aws-docs) |
| Cache | Redis Stack (HNSW, M=64, EF_RUNTIME=300) |
| LLM | Any OpenAI-compatible endpoint (OpenRouter, OpenAI, Anthropic, LMStudio, Ollama, vLLM) |
| Judge | OpenRouter `openai/gpt-oss-120b:nitro` (separate model, avoids self-evaluation bias) |
| CLI | Typer |
| Dashboard | Vanilla HTML/CSS + Chart.js |

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

# --- Option A: Internet-independent (recommended) ---
$env:VECTOR_DB = "qdrant"
pip install pyarrow huggingface_hub
python scripts/setup_vectordb.py --force    # downloads 89k vectors from HuggingFace

# --- Option B: Use Pinecone cloud ---
# $env:VECTOR_DB = "pinecone"  # default

# Run the RAG baseline (200 questions)
cag-lab run --config configs/experiments/rag_baseline.yaml

# Run with semantic cache (500 workload items)
cag-lab run --config configs/experiments/semantic_cache.yaml

# Generate comparison report
python scripts/generate_comparison_report.py --auto
```

## Project Structure

```
cag-lab/
├── configs/
│   ├── experiments/         # YAML experiment definitions
│   └── models/              # Pricing table (editable)
├── data/
│   └── benchmark_sets/      # JSONL Q&A datasets
├── docs/                    # GitHub Pages dashboard
│   ├── index.html           # Landing page
│   ├── rag.html             # RAG deep-dive + SVG architecture
│   ├── cag.html             # CAG deep-dive + research citations
│   ├── report.html          # Auto-generated comparison (Chart.js)
│   └── report.md            # Downloadable markdown report
├── results/
│   ├── jsonl/               # Raw experiment results
│   └── reports/             # Per-experiment markdown reports
├── scripts/
│   ├── generate_comparison_report.py
│   ├── migrate_pinecone_to_qdrant.py
│   ├── rebuild_benchmark.py
│   ├── setup_vectordb.py              # Download vectors from HuggingFace → Qdrant
│   └── export_to_huggingface.py       # Export Qdrant/Pinecone → Parquet → HF Hub
├── src/
│   └── cag_lab/
│       ├── benchmark/       # Dataset, metrics, runner, workload
│       ├── cache/           # Redis HNSW semantic cache
│       ├── embeddings.py    # Shared embedding client
│       ├── generation/      # LLM client + answer generator
│       └── retrieval/       # Qdrant & Pinecone retrievers
├── docker-compose.yml       # Redis Stack + Qdrant
└── pyproject.toml
```

## Design Decisions

### Why GPT-4o-mini is a deliberate choice (no longer used, but the reasoning stands)

We deliberately used the smallest model early on. The point is not to see how well
an LLM can answer from training data — a frontier model would ace questions
without retrieval. We want a model that **must rely on retrieved chunks** so the
benchmark measures retrieval quality, not model memorization.

Current experiments use `nvidia/nemotron-nano-9b-v2` via OpenRouter — a 9B model
that balances reasoning with cost.

### Why threshold 0.96

At 0.92 (default), the semantic cache allowed borderline hits that served wrong
answers for troubleshooting and multi-hop questions. Raising to 0.96 eliminated
false positives entirely (0.0%) while maintaining 73% paraphrase hit rate.
Quality improved from -1% vs RAG to +6%.

### Why Qdrant over Pinecone

Research must be reproducible offline. Cloud-only benchmarks can't be recreated
by collaborators. Qdrant provides identical cosine search with zero external
dependencies. The migration preserves every vector, and `VECTOR_DB` env switching
enables cloud-vs-local latency A/B testing. The full 89,221-vector collection is
published on [HuggingFace Datasets](https://huggingface.co/datasets/mouadja/aws-docs)
so anyone can restore it with a single command — no Pinecone subscription required.

### Why judge models are separate

Self-evaluation bias: if the same model generates and judges answers, it favors
its own outputs. We use `openai/gpt-oss-120b:nitro` via OpenRouter for judging,
routed through `JUDGE_API_KEY` to stay independent of the generation pipeline.

## Pinecone → Qdrant Migration

> **New:** You no longer need Pinecone to populate Qdrant. Run
> `python scripts/setup_vectordb.py` to download the pre-built collection from
> [HuggingFace](https://huggingface.co/datasets/mouadja/aws-docs). The migration
> script below is kept for users who have their own Pinecone index.

### What We Migrated

| From | To |
|---|---|
| Pinecone (cloud, 89,221 vectors) | Qdrant (local, Docker) |
| 512-dim `aws-docs` index | 512-dim `aws-docs` collection |
| Backslash path IDs | Deterministic UUIDs (uuid5 from original) |
| Metadata: content, filePath, chunkIndex, etc. | Exact copy + `_pinecone_id` |

### How It Works

```
┌─────────────┐     list() / fetch()     ┌───────────┐     upsert()     ┌───────────┐
│   Pinecone  │ ───────────────────────→ │ Migration │ ───────────────→ │  Qdrant   │
│ cloud index │ ←── vectors + metadata ─ │ script    │ ←──── UUIDs ──── │  (local)  │
└─────────────┘                          └───────────┘                  └───────────┘
```

1. **List** — Pinecone's `list(prefix="")` paginates through all 89,221 vector IDs
2. **Fetch** — Each batch retrieved with full embeddings and metadata
3. **Transform IDs** — Pinecone's path-style IDs rejected by Qdrant → deterministic UUIDs
4. **Upsert** — All points uploaded to local Qdrant with cosine-distance HNSW index

### Switching Backends

```bash
export VECTOR_DB=qdrant   # local
export VECTOR_DB=pinecone  # cloud (default)
```

## Provider Configuration

The LLM and embedding endpoints are fully configurable via `.env`:

```bash
# OpenRouter (default)
LLM_API_BASE=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...

# OpenAI direct
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sk-...

# Anthropic (OpenAI-compatible endpoint)
LLM_API_BASE=https://api.anthropic.com/v1
LLM_API_KEY=sk-ant-...

# Local LLM (LMStudio / Ollama / vLLM)
LLM_API_BASE=http://localhost:1234/v1
LLM_API_KEY=not-needed
LLM_MODEL=local-model

# Judge model (separate API key for bias avoidance)
JUDGE_API_KEY=sk-or-v1-...
JUDGE_MODEL=openrouter/openai/gpt-oss-120b:nitro

# Embedding (OpenAI text-embedding-3-small at 512d)
EMBED_API_BASE=https://api.openai.com/v1
EMBED_API_KEY=sk-...
EMBED_MODEL=text-embedding-3-small
EMBED_DIMENSIONS=512
```

## Roadmap

This is an active playground — here's what's done and what's coming:

- [x] **Local vector DB** — Qdrant (Docker) replacing Pinecone cloud
- [x] **Direct OpenAI client** — Replaced LiteLLM, clean provider-agnostic routing
- [x] **Multi-provider LLMs** — OpenRouter, OpenAI, Anthropic, LMStudio, Ollama, vLLM
- [x] **Dataset coherence** — Rebuilt 200 questions from the vector DB itself
- [x] **Disaster recovery** — Streaming JSONL, auto-resume, live reports every 10
- [x] **Rich latency metrics** — p25/p50/p75/p95/p99/min/max/mean
- [x] **Dataset sharing** — 89k vectors published to [HuggingFace Datasets](https://huggingface.co/datasets/mouadja/aws-docs), one-command restore
- [ ] **Local SLMs** — Test with Ollama/LM Studio models for zero-API-cost benchmarks
- [ ] **Long-context CAG** — Preload knowledge base into context, cache KV state
- [ ] **Bigger benchmark** — Azure, GCP, Kubernetes documentation domains
- [ ] **Multi-model comparison** — GPT-4o, Claude, Gemini side-by-side
- [ ] **Streaming mode** — Measure time-to-first-token
- [ ] **Eviction policies** — LRU, LFU, score-based cache eviction
- [ ] **Hybrid search** — Dense + sparse (BM25) retrieval

## License

MIT — see [LICENSE](LICENSE). The benchmark dataset is derived from public AWS
documentation.
