"""MNEMONIC CLI — entry point for the `mnemonic` command."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from mnemonic import __version__

app = typer.Typer(
    name="mnemonic",
    help="MNEMONIC: Memory Network Evaluation Metric for Ongoing Natural Interaction and Cognition",
)
console = Console()

# ── Adapter / LLM registry ─────────────────────────────────────

ADAPTERS: dict[str, str] = {
    "engram": "Engram memory system (CLI subprocess)",
}

LLM_PROVIDERS: dict[str, str] = {
    "ollama": "Local models via Ollama API",
    "openai": "OpenAI-compatible API (OpenRouter, etc.)",
}


# ── Factory helpers ─────────────────────────────────────────────


def _build_adapter(name: str, **kwargs):
    """Resolve adapter name to a configured adapter instance."""
    if name == "engram":
        from mnemonic.adapters.engram import EngramAdapter, EngramConfig

        cfg = EngramConfig(
            binary_path=kwargs.get("engram_binary", "engram-mcp"),
            db_path=kwargs.get("db_path", ""),
            ollama_url=kwargs.get("ollama_url", "http://localhost:11434"),
            embed_model=kwargs.get("embed_model", "mxbai-embed-large"),
            top_k=kwargs.get("top_k", 20),
            build_relations=kwargs.get("build_relations", False),
            rich_recall=kwargs.get("rich_recall", False),
            chain_depth=kwargs.get("chain_depth", 3),
        )
        return EngramAdapter(config=cfg)

    console.print(f"[red]Error:[/red] Unknown adapter: {name}")
    console.print(f"Available: {', '.join(ADAPTERS)}")
    raise typer.Exit(1)


def _build_llm(name: str, **kwargs):
    """Resolve LLM provider name to a configured provider instance."""
    if name == "openai":
        from mnemonic.llm.openai import OpenAIConfig, OpenAIProvider

        cfg = OpenAIConfig(
            base_url=kwargs.get("api_base", "https://openrouter.ai/api/v1"),
            api_key=kwargs.get("api_key", ""),
            model=kwargs.get("model", "openai/gpt-4o-mini"),
            temperature=kwargs.get("temperature", 0.0),
            max_tokens=kwargs.get("max_tokens", 10),
        )
        return OpenAIProvider(config=cfg)

    if name == "ollama":
        from mnemonic.llm.ollama import OllamaConfig, OllamaProvider

        cfg = OllamaConfig(
            base_url=kwargs.get("ollama_url", "http://localhost:11434"),
            model=kwargs.get("model", "gemma3:27b"),
            temperature=kwargs.get("temperature", 0.0),
        )
        return OllamaProvider(config=cfg)

    console.print(f"[red]Error:[/red] Unknown LLM provider: {name}")
    console.print(f"Available: {', '.join(LLM_PROVIDERS)}")
    raise typer.Exit(1)


# ── Commands ────────────────────────────────────────────────────


@app.command()
def run(
    adapter: str = typer.Option(..., "--adapter", "-a", help="Memory system adapter name (e.g. engram)"),
    llm: str = typer.Option(..., "--llm", "-l", help="LLM provider name (e.g. openai, ollama)"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to dataset JSON/JSONL file"),
    model: str = typer.Option("openai/gpt-4o-mini", "--model", "-m", help="Model name for the LLM provider"),
    top_k: int = typer.Option(20, "--top-k", "-k", help="Number of memories to retrieve per query"),
    prompt_version: str = typer.Option("v1", "--prompt-version", "-p", help="Prompt template version (v1=basic, v2=enriched metadata)"),
    max_questions: Optional[int] = typer.Option(None, "--max-questions", "-n", help="Limit number of questions (for quick validation)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENROUTER_API_KEY", help="API key (or set OPENROUTER_API_KEY env var)"),
    api_base: str = typer.Option("https://openrouter.ai/api/v1", "--api-base", help="Base URL for OpenAI-compatible API"),
    engram_binary: str = typer.Option("engram-mcp", "--engram-binary", help="Path to engram-mcp binary"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url", help="Ollama server URL"),
    embed_model: str = typer.Option("mxbai-embed-large", "--embed-model", help="Embedding model for engram"),
    concurrency: int = typer.Option(10, "--concurrency", "-c", help="Number of concurrent streams for ingest and answering"),
    build_relations: bool = typer.Option(False, "--build-relations", help="Build Precedes + semantic RelatedTo relations during ingest"),
    rich_recall: bool = typer.Option(False, "--rich-recall", help="Use rich recall output with dates and corroboration metadata"),
    chain_depth: int = typer.Option(0, "--chain-depth", help="Chain recall depth (0=disabled, 3=default)"),
    use_intent: bool = typer.Option(False, "--use-intent", help="Send query intent hints (temporal, factual, etc.) per question category"),
) -> None:
    """Run the benchmark suite against a memory system."""
    from mnemonic.prompts import PROMPT_VERSIONS

    if not dataset.exists():
        console.print(f"[red]Error:[/red] Dataset not found: {dataset}")
        raise typer.Exit(1)

    if prompt_version not in PROMPT_VERSIONS:
        console.print(f"[red]Error:[/red] Unknown prompt version: {prompt_version}")
        console.print(f"Available: {', '.join(sorted(PROMPT_VERSIONS))}")
        raise typer.Exit(1)

    if llm == "openai" and not api_key:
        console.print("[red]Error:[/red] --api-key or OPENROUTER_API_KEY env var required for openai provider")
        raise typer.Exit(1)

    # Build components
    adapter_instance = _build_adapter(
        adapter,
        engram_binary=engram_binary,
        ollama_url=ollama_url,
        embed_model=embed_model,
        top_k=top_k,
        build_relations=build_relations,
        rich_recall=rich_recall,
        chain_depth=chain_depth,
    )

    llm_instance = _build_llm(
        llm,
        model=model,
        api_key=api_key or "",
        api_base=api_base,
        ollama_url=ollama_url,
    )

    # Load dataset and run
    from mnemonic.suites.base import BaseSuite, load_locomo_dataset
    from mnemonic.scoring.scorer import Scorer

    ds = load_locomo_dataset(dataset)
    suite = BaseSuite(
        adapter=adapter_instance,
        llm_provider=llm_instance,
        prompt_version=prompt_version,
        concurrency=concurrency,
        top_k=top_k,
        use_intent=use_intent,
    )

    result = asyncio.run(suite.run(ds, max_questions=max_questions))
    Scorer.print_report(result)


@app.command()
def list_adapters() -> None:
    """List available memory system adapters and LLM providers."""
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
