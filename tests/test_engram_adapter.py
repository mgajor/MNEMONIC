"""Tests for the Engram adapter — parser, config, and structure."""

import pytest

from mnemonic.adapters.engram import (
    EngramAdapter,
    EngramConfig,
    parse_recall_output,
)


# ── Config tests ────────────────────────────────────────────────


def test_default_config_creates_temp_db() -> None:
    cfg = EngramConfig()
    assert cfg.db_path.endswith(".db")
    assert "mnemonic_engram_" in cfg.db_path


def test_custom_config() -> None:
    cfg = EngramConfig(
        binary_path="/usr/local/bin/engram-mcp",
        db_path="/tmp/test.db",
        ollama_url="http://gpu-box:11434",
        embed_model="nomic-embed-text",
        top_k=10,
    )
    assert cfg.binary_path == "/usr/local/bin/engram-mcp"
    assert cfg.db_path == "/tmp/test.db"
    assert cfg.top_k == 10


# ── Recall parser tests ────────────────────────────────────────

# Matches actual engram-mcp recall output format:
# [score|certainty|source|kind] content
SAMPLE_RECALL_OUTPUT = """\
[0.87|certain|Direct|Semantic] Alice works at Acme Corp as a senior engineer
[0.74|likely|Consolidated|Episodic] Bob mentioned he moved to Seattle last month
[0.61|vague|Inferred|Procedural] They discussed the quarterly review process
"""


def test_parse_recall_output_basic() -> None:
    results = parse_recall_output(SAMPLE_RECALL_OUTPUT)
    assert len(results) == 3

    assert results[0].content == "Alice works at Acme Corp as a senior engineer"
    assert results[0].score == 0.87
    assert results[0].certainty == "certain"
    assert results[0].kind == "semantic"
    assert results[0].source == "Direct"

    assert results[1].content == "Bob mentioned he moved to Seattle last month"
    assert results[1].score == 0.74
    assert results[1].certainty == "likely"
    assert results[1].source == "Consolidated"

    assert results[2].content == "They discussed the quarterly review process"
    assert results[2].certainty == "vague"
    assert results[2].kind == "procedural"
    assert results[2].source == "Inferred"


def test_parse_recall_output_empty() -> None:
    assert parse_recall_output("") == []
    assert parse_recall_output("\n\n") == []


def test_parse_recall_output_single_line() -> None:
    line = "[0.95|certain|Direct|Semantic] Hello world"
    results = parse_recall_output(line)
    assert len(results) == 1
    assert results[0].content == "Hello world"
    assert results[0].score == 0.95


def test_parse_recall_with_brackets_in_content() -> None:
    """Content after the metadata bracket should be captured as-is."""
    line = "[0.80|likely|Direct|Episodic] Alice said [laughing] that she loved it"
    results = parse_recall_output(line)
    assert len(results) == 1
    assert results[0].content == "Alice said [laughing] that she loved it"
    assert results[0].score == 0.80
    assert results[0].certainty == "likely"


def test_parse_recall_skips_non_matching_lines() -> None:
    """Non-recall lines (e.g. log output) should be silently skipped."""
    output = """\
some random log line
[0.70|certain|Direct|Semantic] Real memory content
another junk line
"""
    results = parse_recall_output(output)
    assert len(results) == 1
    assert results[0].content == "Real memory content"


# ── Adapter structure tests ─────────────────────────────────────


def test_adapter_has_required_attributes() -> None:
    adapter = EngramAdapter()
    assert adapter.name == "engram"
    assert adapter.version == "0.1.0"


def test_adapter_base_args() -> None:
    cfg = EngramConfig(
        binary_path="/path/to/engram-mcp",
        db_path="/tmp/test.db",
        ollama_url="http://localhost:11434",
        embed_model="mxbai-embed-large",
    )
    adapter = EngramAdapter(config=cfg)
    args = adapter._base_args("conv-42")

    assert args[0] == "/path/to/engram-mcp"
    assert "--db-path" in args
    assert "/tmp/test.db" in args
    assert "--namespace" in args
    assert "conv-42" in args
    assert "--ollama-url" in args
    assert "--embed-model" in args


@pytest.mark.asyncio
async def test_adapter_reset_creates_new_db_path() -> None:
    adapter = EngramAdapter()
    original_path = adapter.config.db_path
    adapter._namespaces.add("test-ns")

    await adapter.reset()

    assert adapter.config.db_path != original_path
    assert adapter.config.db_path.endswith(".db")
    assert len(adapter._namespaces) == 0
