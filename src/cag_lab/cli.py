from importlib.metadata import version

import typer

from cag_lab.config import get_openrouter_model

app = typer.Typer(no_args_is_help=True)

_OPENROUTER_MODEL = f"openrouter/{get_openrouter_model()}"

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
    index: str = typer.Option("aws-docs", "--index", "-i", help="Pinecone index name"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to return"),
    model: str = typer.Option(
        "text-embedding-3-small", "--model", "-m", help="Embedding model name"
    ),
    dimensions: int = typer.Option(
        512, "--dimensions", "-d", help="Embedding dimensions"
    ),
) -> None:
    """Retrieve chunks from a Pinecone index."""
    from cag_lab.retrieval.pinecone_retriever import Retriever

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
    index: str = typer.Option("aws-docs", "--index", "-i", help="Pinecone index name"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    emb_model: str = typer.Option("text-embedding-3-small", "--emb-model", help="Embedding model name"),
    dimensions: int = typer.Option(512, "--dimensions", "-d", help="Embedding dimensions"),
    llm_model: str = typer.Option(
        _OPENROUTER_MODEL, "--model", "-m", help="LLM model for generation"
    ),
    api_base: str | None = typer.Option(None, "--api-base", help="Custom endpoint for OpenAI-compatible APIs (LMStudio, vLLM)"),
) -> None:
    """Answer a question using RAG over a Pinecone index."""
    from cag_lab.generation.answer_generator import generate_answer
    from cag_lab.retrieval.pinecone_retriever import Retriever

    retriever = Retriever(
        index_name=index,
        embedding_model=emb_model,
        embedding_dimensions=dimensions,
        top_k=top_k,
    )
    chunks = retriever.retrieve(query)
    result = generate_answer(query, chunks, model=llm_model, api_base=api_base)

    typer.echo(result.answer)
    typer.echo(f"\nSources: {result.sources}")
    typer.echo(f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
    
    for i, chunk in enumerate(chunks, 1):
        preview = chunk.text[:200].replace("\n", " ")
        typer.echo(f"\n--- Chunk {i} (score: {chunk.score:.4f}) ---")
        if i == 1:
            typer.echo(f"Metadata keys: {list(chunk.source.keys())}")
        typer.echo(preview)




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