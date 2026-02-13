"""Abstract base class for LLM answer-generation providers."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Contract for LLM providers used in answer generation.

    The LLM provider is intentionally decoupled from the memory adapter.
    The same memory system can produce very different benchmark scores
    depending on which LLM interprets the retrieved context — this
    separation lets us measure that variance directly.
    """

    name: str

    @abstractmethod
    async def generate_answer(
        self, context: str, question: str, choices: list[str]
    ) -> str:
        """Given retrieved context, a question, and choices, return an answer letter.

        Args:
            context: Formatted string of retrieved memory fragments.
            question: The benchmark question text.
            choices: List of answer options, e.g. ["A) Acme Corp", "B) Globex", ...].

        Returns:
            A single uppercase letter (e.g. "A", "B") representing the chosen answer.
        """
