# MNEMONIC

**Memory Network Evaluation Metric for Ongoing Natural Interaction and Cognition**

A standalone benchmark for evaluating long-term conversational memory systems. MNEMONIC measures how well AI memory backends retain, recall, and reason over information across extended multi-turn conversations.

## Features

- **Adapter interface** — plug in any memory system (vector stores, knowledge graphs, hybrid systems) via a unified adapter API
- **Test suites** — structured evaluation scenarios covering retention, recall accuracy, temporal reasoning, contradiction detection, and more
- **Scoring engine** — quantitative metrics with weighted dimensions
- **Leaderboard** — compare memory systems side-by-side
- **CLI-first** — run benchmarks, inspect results, and manage adapters from the terminal

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --group dev
```

## Usage

```bash
# Show available commands
mnemonic --help

# Print version
mnemonic version

# Run the benchmark (coming soon)
mnemonic run

# List available adapters (coming soon)
mnemonic list-adapters

# View results (coming soon)
mnemonic results
```

## Development

```bash
# Install dependencies (including dev tools)
uv sync --group dev

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

## Project Structure

```
mnemonic_benchmark/
├── pyproject.toml            # Build config (uv/hatchling)
├── mnemonic/
│   ├── __init__.py           # Package version
│   ├── cli.py                # Typer CLI entry point
│   ├── adapters/             # Memory system adapter interface
│   ├── suites/               # Benchmark test suites
│   ├── scoring/              # Scoring engine
│   └── results/              # Leaderboard & reporting
├── tests/                    # pytest test suite
└── data/                     # Conversation datasets (future)
```

## License

TBD
