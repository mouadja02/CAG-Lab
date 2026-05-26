from importlib.metadata import version

import typer

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    """cag-lab — benchmark lab for RAG, semantic caching, and long-context generation."""


@app.command(name="version")
def version_cmd() -> None:
    """Print the package version."""
    typer.echo(version("cag_lab"))
