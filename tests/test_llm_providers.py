"""Tests for LLM providers — config, answer extraction, request building."""

import pytest

from mnemonic.llm.openai import (
    OpenAIConfig,
    OpenAIProvider,
    _extract_answer_letter,
    _is_reasoning_model,
)
from mnemonic.llm.ollama import OllamaConfig, OllamaProvider


# ── Answer letter extraction ────────────────────────────────────


def test_extract_single_letter() -> None:
    assert _extract_answer_letter("A") == "A"
    assert _extract_answer_letter("b") == "B"
    assert _extract_answer_letter(" C ") == "C"


def test_extract_letter_with_paren() -> None:
    assert _extract_answer_letter("A)") == "A"
    assert _extract_answer_letter("B.") == "B"


def test_extract_answer_phrase() -> None:
    assert _extract_answer_letter("The answer is C") == "C"
    assert _extract_answer_letter("Answer: D") == "D"
    assert _extract_answer_letter("The correct choice is B") == "B"


def test_extract_standalone_letter() -> None:
    assert _extract_answer_letter("I think it's A based on the context") == "A"


def test_extract_empty() -> None:
    assert _extract_answer_letter("") == "?"


# ── Reasoning model detection ──────────────────────────────────


def test_reasoning_model_detection() -> None:
    assert _is_reasoning_model("o3") is True
    assert _is_reasoning_model("openai/o3-mini") is True
    assert _is_reasoning_model("gpt-5-mini") is True
    assert _is_reasoning_model("openai/gpt-4o-mini") is False
    assert _is_reasoning_model("anthropic/claude-sonnet-4-5") is False


# ── OpenAI config ──────────────────────────────────────────────


def test_openai_default_config() -> None:
    cfg = OpenAIConfig()
    assert cfg.base_url == "https://openrouter.ai/api/v1"
    assert cfg.model == "openai/gpt-4o-mini"
    assert cfg.temperature == 0.0
    assert cfg.max_tokens == 10


def test_openai_reasoning_model_bumps_max_tokens() -> None:
    cfg = OpenAIConfig(model="openai/o3-mini")
    assert cfg.max_tokens >= 2000


def test_openai_provider_name() -> None:
    cfg = OpenAIConfig(model="google/gemini-3-flash")
    provider = OpenAIProvider(config=cfg)
    assert provider.name == "google/gemini-3-flash"


# ── Ollama config ──────────────────────────────────────────────


def test_ollama_default_config() -> None:
    cfg = OllamaConfig()
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.model == "gemma3:27b"
    assert cfg.temperature == 0.0


def test_ollama_provider_name() -> None:
    cfg = OllamaConfig(model="llama3:8b")
    provider = OllamaProvider(config=cfg)
    assert provider.name == "ollama/llama3:8b"
