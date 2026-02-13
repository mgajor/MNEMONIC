"""Tests for prompt template formatting."""

from mnemonic.prompts import build_prompt, format_context_v1, format_context_v2
from mnemonic.types import RecallResult


SAMPLE_MEMORIES = [
    RecallResult(
        content="Alice works at Acme Corp",
        score=0.87,
        certainty="certain",
        kind="semantic",
        source="Direct",
        memory_id="a1b2c3d4",
    ),
    RecallResult(
        content="Bob moved to Seattle",
        score=0.74,
        certainty="likely",
        kind="episodic",
        source="Consolidated",
        memory_id="e5f6a7b8",
    ),
]


def test_format_context_v1() -> None:
    ctx = format_context_v1(SAMPLE_MEMORIES)
    assert "[1] Alice works at Acme Corp" in ctx
    assert "[2] Bob moved to Seattle" in ctx
    # v1 should NOT include metadata
    assert "certain" not in ctx
    assert "semantic" not in ctx


def test_format_context_v2() -> None:
    ctx = format_context_v2(SAMPLE_MEMORIES)
    # v2 should include metadata tags
    assert "[1]" in ctx
    assert "certain" in ctx
    assert "direct" in ctx
    assert "semantic" in ctx
    assert "Alice works at Acme Corp" in ctx

    assert "[2]" in ctx
    assert "likely" in ctx
    assert "Bob moved to Seattle" in ctx


def test_build_prompt_v1() -> None:
    prompt = build_prompt(
        memories=SAMPLE_MEMORIES,
        question="Where does Alice work?",
        choices=["A) Acme Corp", "B) Globex"],
        prompt_version="v1",
    )
    assert "Where does Alice work?" in prompt
    assert "A) Acme Corp" in prompt
    assert "Respond with ONLY the letter" in prompt
    assert "[1] Alice works at Acme Corp" in prompt
    # v1 header
    assert "timestamps in brackets" in prompt


def test_build_prompt_v2() -> None:
    prompt = build_prompt(
        memories=SAMPLE_MEMORIES,
        question="Where does Alice work?",
        choices=["A) Acme Corp", "B) Globex"],
        prompt_version="v2",
    )
    assert "Where does Alice work?" in prompt
    assert "Certainty tags" in prompt
    assert "certain" in prompt
    assert "direct" in prompt


def test_build_prompt_defaults_to_v1() -> None:
    prompt = build_prompt(
        memories=SAMPLE_MEMORIES,
        question="Test?",
        choices=["A) Yes"],
    )
    assert "timestamps in brackets" in prompt


def test_build_prompt_empty_memories() -> None:
    prompt = build_prompt(
        memories=[],
        question="No context?",
        choices=["A) Unknown"],
        prompt_version="v1",
    )
    assert "No context?" in prompt
    assert "A) Unknown" in prompt
