"""Tests for the Engram adapter — parser, config, and structure."""

import pytest

from mnemonic.adapters.engram import (
    EngramAdapter,
    EngramConfig,
    parse_recall_output,
    _parse_recall_line_fallback,
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


SAMPLE_RECALL_OUTPUT = """\
1. [0.87] (certain) (Semantic) Alice works at Acme Corp as a senior engineer (id: a1b2c3d4, source: Direct)
2. [0.74] (likely) (Episodic) Bob mentioned he moved to Seattle last month (id: e5f6g7h8, source: Consolidated)
3. [0.61] (vague) (Procedural) They discussed the quarterly review process (id: i9j0k1l2, source: Inferred)
"""


def test_parse_recall_output_basic() -> None:
    results = parse_recall_output(SAMPLE_RECALL_OUTPUT)
    assert len(results) == 3
    assert results[0] == "Alice works at Acme Corp as a senior engineer"
    assert results[1] == "Bob mentioned he moved to Seattle last month"
    assert results[2] == "They discussed the quarterly review process"


def test_parse_recall_output_empty() -> None:
    assert parse_recall_output("") == []
    assert parse_recall_output("\n\n") == []


def test_parse_recall_output_single_line() -> None:
    line = "1. [0.95] (certain) (Semantic) Hello world (id: abcd1234, source: Direct)"
    results = parse_recall_output(line)
    assert len(results) == 1
    assert results[0] == "Hello world"


def test_parse_recall_with_parentheses_in_content() -> None:
    """Content with parens should be handled by the fallback parser."""
    line = "1. [0.80] (likely) (Episodic) Alice said (laughing) that she loved it (id: abcd1234, source: Direct)"
    results = parse_recall_output(line)
    assert len(results) == 1
    assert "Alice said (laughing) that she loved it" in results[0]


# ── Fallback parser tests ──────────────────────────────────────


def test_fallback_parser_with_parens() -> None:
    line = "1. [0.80] (likely) (Episodic) Alice said (laughing) that she loved it (id: abcd1234, source: Direct)"
    result = _parse_recall_line_fallback(line)
    assert result is not None
    assert "Alice said (laughing) that she loved it" in result


def test_fallback_parser_no_id_marker() -> None:
    assert _parse_recall_line_fallback("just some random text") is None


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
