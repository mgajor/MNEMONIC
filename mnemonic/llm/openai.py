"""OpenAI-compatible LLM provider — works with OpenRouter, OpenAI, and any compatible API."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import httpx

from mnemonic.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)

# Models that need large max_tokens or they return empty content
REASONING_MODELS = {"o3", "o3-mini", "o4-mini", "gpt-5-mini", "kimi-k2.5"}


def _is_reasoning_model(model: str) -> bool:
    """Check if the model is a reasoning model that needs large max_tokens."""
    model_lower = model.lower()
    return any(rm in model_lower for rm in REASONING_MODELS)


def _extract_answer_letter(text: str) -> str:
    """Extract a single answer letter from LLM response text.

    Handles responses like "A", "A)", "The answer is B", "Answer: C", etc.
    """
    text = text.strip()

    # Direct single letter
    if len(text) == 1 and text.isalpha():
        return text.upper()

    # Letter followed by ) or .
    match = re.match(r"^([A-Za-z])[).]", text)
    if match:
        return match.group(1).upper()

    # "Answer: X" or "The answer is X"
    match = re.search(r"(?:answer|choice)\s*(?:is|:)\s*([A-Za-z])", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Last resort: find a standalone capital letter A-J, skipping "I" (pronoun)
    for match in re.finditer(r"\b([A-HJ])\b", text):
        return match.group(1)

    # If we really can't find anything, return the first character
    return text[0].upper() if text else "?"


@dataclass
class OpenAIConfig:
    """Configuration for an OpenAI-compatible LLM provider."""

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 10
    timeout: float = 60.0

    def __post_init__(self) -> None:
        # Reasoning models need large max_tokens or they return empty
        if _is_reasoning_model(self.model) and self.max_tokens < 2000:
            self.max_tokens = 2000


class OpenAIProvider(BaseLLMProvider):
    """LLM provider for any OpenAI-compatible API (OpenRouter, OpenAI, etc.)."""

    def __init__(self, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout,
        )

    @property
    def name(self) -> str:
        return self.config.model

    async def generate_answer(
        self, context: str, question: str, choices: list[str]
    ) -> str:
        """Send prompt to OpenAI-compatible API and extract answer letter."""
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": context}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        if not content:
            logger.warning("Empty response from %s", self.config.model)
            return "?"

        return _extract_answer_letter(content)

    async def close(self) -> None:
        await self._client.aclose()
