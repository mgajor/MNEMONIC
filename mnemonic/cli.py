"""MNEMONIC CLI — entry point for the `mnemonic` command."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from mnemonic import __version__

app = typer.Typer(
    name="mnemonic",
    help="MNEMONIC: Memory Network Evaluation Metric for Ongoing Natural Interaction and Cognition",
)
console = Console()

# ── Adapter / LLM registry (will grow as adapters are added) ──

ADAPTERS: dict[str, str] = {
    "engram": "Engram memory system (CLI subprocess)",
}

LLM_PROVIDERS: dict[str, str] = {
    "ollama": "Local models via Ollama API",
    "openai": "OpenAI-compatible API (OpenRouter, etc.)",
}


@app.command()
def run(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Memory system adapter name"),
    llm: str = typer.Option(..., "--llm", "-l", help="LLM provider name"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to dataset JSON file"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Number of memories to retrieve per query"),
) -> None:
    """Run the benchmark suite against a memory system."""
    if not dataset.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {dataset}")
        raise typer.Exit(1)

    console.print(
        f"\n[bold]MNEMONIC Benchmark[/bold]"
        f"\n  Adapter:  [cyan]{adapter}[/cyan]"
        f"\n  LLM:      [cyan]{llm}[/cyan]"
        f"\n  Dataset:  [cyan]{dataset}[/cyan]"
        f"\n  top_k:    {top_k}\n"
    )

    # TODO: resolve adapter and llm_provider from registry, load dataset,
    #       instantiate BaseSuite, and call suite.run()
    console.print(
        "[yellow]Adapter and LLM provider loading not yet wired.[/yellow]\n"
        "Once concrete adapters are registered, this command will:\n"
        "  1. Load the dataset\n"
        "  2. Instantiate the adapter and LLM provider\n"
        "  3. Run ingest -> recall -> answer -> score\n"
        "  4. Print the results report\n"
    )


@app.command()
def list_adapters() -> None:
    """List available memory system adapters."""
    table = Table(title="Available Adapters")
    table.add_column("Name", style="cyan")
    table.add_column("Description")

    for name, desc in ADAPTERS.items():
        table.add_row(name, desc)

    console.print(table)

    console.print()
    table2 = Table(title="Available LLM Providers")
    table2.add_column("Name", style="cyan")
    table2.add_column("Description")

    for name, desc in LLM_PROVIDERS.items():
        table2.add_row(name, desc)

    console.print(table2)


@app.command()
def results() -> None:
    """Show benchmark results."""
    console.print("[yellow]Not yet implemented:[/yellow] results viewer")
    console.print("Will display saved benchmark results and leaderboard comparisons.")


@app.command()
def version() -> None:
    """Print the current version."""
    typer.echo(f"mnemonic {__version__}")
