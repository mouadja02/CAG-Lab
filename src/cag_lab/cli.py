from importlib.metadata import version
from functools import lru_cache

import typer

app = typer.Typer(no_args_is_help=True)


@lru_cache(maxsize=1)
def _default_llm_model() -> str:
    from cag_lab.config import get_llm_model

    return get_llm_model()


@app.callback()
def main() -> None:
    """cag-lab — benchmark lab for RAG, semantic caching, and long-context generation."""


@app.command(name="version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(version("cag_lab"))


@app.command(name="retrieve")
def retrieve(
    query: str,
    index: str = typer.Option(
        "aws-docs", "--index", "-i", help="Vector DB index or collection name"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to return"),
    model: str = typer.Option(
        "text-embedding-3-small", "--model", "-m", help="Embedding model name"
    ),
    dimensions: int = typer.Option(
        512, "--dimensions", "-d", help="Embedding dimensions"
    ),
    embed_api_base: str | None = typer.Option(
        None,
        "--embed-api-base",
        help="Custom embedding endpoint (LMStudio, Ollama, vLLM)",
    ),
) -> None:
    """Retrieve chunks from the vector DB."""
    from cag_lab.retrieval import Retriever

    retriever = Retriever(
        index_name=index,
        embedding_model=model,
        embedding_dimensions=dimensions,
        top_k=top_k,
        embed_api_base=embed_api_base,
    )
    from cag_lab.retrieval import Retriever

    retriever = Retriever(
        index_name=index,
        embedding_model=model,
        embedding_dimensions=dimensions,
        top_k=top_k,
    )
    chunks = retriever.retrieve(query)

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:200].replace("\n", " ")
        typer.echo(f"\n--- Chunk {i} (score: {chunk.score:.4f}) ---")
        if i == 1:
            typer.echo(f"Metadata keys: {list(chunk.source.keys())}")
        typer.echo(preview)


@app.command(name="ask")
def ask(
    query: str,
    index: str = typer.Option(
        "aws-docs", "--index", "-i", help="Vector DB index or collection name"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    emb_model: str = typer.Option(
        "text-embedding-3-small", "--emb-model", help="Embedding model name"
    ),
    dimensions: int = typer.Option(
        512, "--dimensions", "-d", help="Embedding dimensions"
    ),
    llm_model: str = typer.Option(
        _default_llm_model(), "--model", "-m", help="LLM model for generation"
    ),
    api_base: str | None = typer.Option(
        None,
        "--api-base",
        help="Custom LLM endpoint (OpenRouter, LMStudio, Ollama, vLLM, OpenAI, Anthropic)",
    ),
    embed_api_base: str | None = typer.Option(
        None,
        "--embed-api-base",
        help="Custom embedding endpoint (LMStudio, Ollama, vLLM)",
    ),
) -> None:
    """Answer a question using RAG over the vector DB."""
    from cag_lab.generation.answer_generator import generate_answer
    from cag_lab.retrieval import Retriever

    retriever = Retriever(
        index_name=index,
        embedding_model=emb_model,
        embedding_dimensions=dimensions,
        top_k=top_k,
        embed_api_base=embed_api_base,
    )
    chunks = retriever.retrieve(query)
    result = generate_answer(query, chunks, model=llm_model, api_base=api_base)

    typer.echo(result.answer)
    typer.echo(f"\nSources: {result.sources}")
    typer.echo(
        f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion"
    )

    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:200].replace("\n", " ")
        typer.echo(f"\n--- Chunk {i} (score: {chunk.score:.4f}) ---")
        if i == 1:
            typer.echo(f"Metadata keys: {list(chunk.source.keys())}")
        typer.echo(preview)


@app.command(name="run")
def run(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to experiment YAML file"
    ),
) -> None:
    """Run a benchmark experiment defined in a YAML config file."""
    from cag_lab.benchmark.runner import run_experiment

    run_experiment(config)


@app.command(name="sweep")
def sweep(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to experiment YAML file"
    ),
    seeds: str = typer.Option(
        "42",
        "--seeds",
        "-s",
        help="Comma-separated list of random seeds for multi-run (e.g. 42,123,456)",
    ),
    thresholds: str = typer.Option(
        "",
        "--thresholds",
        "-t",
        help="Comma-separated cache thresholds to sweep (e.g. 0.85,0.90,0.92,0.95)",
    ),
    mixes: str = typer.Option(
        "",
        "--mixes",
        "-m",
        help='Workload mixes as semicolon-separated r,p,n triples (e.g. "0.4,0.2,0.4;0.2,0.2,0.6;0.1,0.1,0.8")',
    ),
) -> None:
    """Run a parameter sweep over seeds, thresholds, and/or workload mixes.

    Examples:
        cag-lab sweep -c configs/experiments/semantic_cache.yaml -s 42,123,456
        cag-lab sweep -c configs/experiments/semantic_cache.yaml -t 0.85,0.90,0.92,0.95
        cag-lab sweep -c configs/experiments/semantic_cache.yaml -s 42,123 -t 0.90,0.95
        cag-lab sweep -c configs/experiments/semantic_cache.yaml -m "0.4,0.2,0.4;0.1,0.1,0.8"
    """
    from cag_lab.benchmark.runner import run_sweep

    parsed_seeds = [int(s.strip()) for s in seeds.split(",") if s.strip()]

    parsed_thresholds = None
    if thresholds.strip():
        parsed_thresholds = [
            float(t.strip()) for t in thresholds.split(",") if t.strip()
        ]

    parsed_mixes = None
    if mixes.strip():
        parsed_mixes = []
        for mix_str in mixes.split(";"):
            parts = [float(x.strip()) for x in mix_str.split(",") if x.strip()]
            if len(parts) != 3:
                typer.echo(
                    f"Error: each mix must have 3 values (repeat,paraphrase,new), got {parts}",
                    err=True,
                )
                raise typer.Exit(1)
            parsed_mixes.append(
                {
                    "repeated_query_rate": parts[0],
                    "paraphrase_rate": parts[1],
                    "new_query_rate": parts[2],
                }
            )

    summaries = run_sweep(
        config,
        seeds=parsed_seeds or None,
        thresholds=parsed_thresholds,
        workload_mixes=parsed_mixes,
    )

    typer.echo(f"\nSweep complete: {len(summaries)} configuration(s) evaluated.")
    for s in summaries:
        typer.echo(f"  {s['run_label']}")


@app.command(name="dataset")
def dataset_cmd(
    action: str = typer.Argument(..., help="Action: validate"),
    path: str = typer.Argument(..., help="Path to JSONL file"),
) -> None:
    """Dataset management commands."""
    if action != "validate":
        typer.echo(f"Unknown action: {action}. Valid actions: validate", err=True)
        raise typer.Exit(1)

    from collections import Counter

    from cag_lab.benchmark.dataset import load_dataset

    questions = load_dataset(path)
    typer.echo(f"Valid questions: {len(questions)}")
    type_counts = Counter(q.query_type.value for q in questions)
    typer.echo("Breakdown by query_type:")
    for qtype, count in sorted(type_counts.items()):
        typer.echo(f"  {qtype}: {count}")
