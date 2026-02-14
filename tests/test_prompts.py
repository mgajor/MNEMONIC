"""Tests for prompt template formatting."""

from mnemonic.prompts import build_prompt, format_context_v1, format_context_v2, _format_choices
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
    # v2 should include certainty metadata
    assert "[1]" in ctx
    assert "certain" in ctx
    # "Direct" source is omitted (default) — only non-direct sources shown
    assert "Alice works at Acme Corp" in ctx

    assert "[2]" in ctx
    assert "likely" in ctx
    assert "consolidated" in ctx  # non-direct source shown
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
    assert "dates in brackets" in prompt
    assert "temporal order" in prompt
    assert "certain" in prompt


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


def test_format_choices_adds_prefixes() -> None:
    """Plain text choices should get letter prefixes added."""
    result = _format_choices(["20 May 2023", "10 May 2023", "6 May 2023"])
    assert "A) 20 May 2023" in result
    assert "B) 10 May 2023" in result
    assert "C) 6 May 2023" in result


def test_format_choices_preserves_existing_prefixes() -> None:
    """Choices already prefixed with letters should be left as-is."""
    result = _format_choices(["A) Acme Corp", "B) Globex"])
    assert "A) Acme Corp" in result
    assert "B) Globex" in result
    # Should NOT double-prefix
    assert "A) A)" not in result


def test_build_prompt_plain_choices() -> None:
    """build_prompt should auto-prefix plain-text choices."""
    prompt = build_prompt(
        memories=SAMPLE_MEMORIES,
        question="When did it happen?",
        choices=["May 2023", "June 2023", "July 2023"],
        prompt_version="v1",
    )
    assert "A) May 2023" in prompt
    assert "B) June 2023" in prompt
    assert "C) July 2023" in prompt
