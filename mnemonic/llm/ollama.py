"""Ollama LLM provider — local model inference."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from mnemonic.llm.base import BaseLLMProvider
from mnemonic.llm.openai import _extract_answer_letter

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """Configuration for the Ollama local LLM provider."""

    base_url: str = "http://localhost:11434"
    model: str = "gemma3:27b"
    temperature: float = 0.0
    num_predict: int = 10
    timeout: float = 120.0


class OllamaProvider(BaseLLMProvider):
    """LLM provider for locally running Ollama models."""

    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    @property
    def name(self) -> str:
        return f"ollama/{self.config.model}"

    async def generate_answer(
        self, context: str, question: str, choices: list[str]
    ) -> str:
        """Send prompt to Ollama /api/generate and extract answer letter."""
        payload = {
            "model": self.config.model,
            "prompt": context,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        }

        response = await self._client.post("/api/generate", json=payload)
        response.raise_for_status()

        data = response.json()
        content = data.get("response", "")

        if not content:
            logger.warning("Empty response from ollama/%s", self.config.model)
            return "?"

        return _extract_answer_letter(content)

    async def close(self) -> None:
        await self._client.aclose()
