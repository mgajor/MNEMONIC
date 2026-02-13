import typer

from mnemonic import __version__

app = typer.Typer(
    name="mnemonic",
    help="MNEMONIC: Memory Network Evaluation Metric for Ongoing Natural Interaction and Cognition",
)


@app.command()
def run() -> None:
    """Run the benchmark suite."""
    typer.echo("Not yet implemented: run")


@app.command()
def list_adapters() -> None:
    """List available memory system adapters."""
    typer.echo("Not yet implemented: list-adapters")


@app.command()
def results() -> None:
    """Show benchmark results."""
    typer.echo("Not yet implemented: results")


@app.command()
def version() -> None:
    """Print the current version."""
    typer.echo(f"mnemonic {__version__}")
