"""Abstract base class for memory system adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mnemonic.types import Conversation, RecallResult


class BaseAdapter(ABC):
    """Contract that all memory system adapters must implement.

    An adapter wraps a memory backend (engram, Mem0, Memobase, Zep, etc.)
    and exposes a uniform interface for the benchmark runner: ingest
    conversations, recall relevant context for a query, reset state,
    and report stats.
    """

    name: str
    version: str

    @abstractmethod
    async def ingest(
        self, conversation: Conversation, concurrency: int = 1
    ) -> dict:
        """Ingest a full conversation into the memory system.

        Args:
            conversation: A Conversation with id, messages, and optional timestamps.
            concurrency: Max concurrent operations (default 1 = sequential).

        Returns:
            Dict with ingest metadata (e.g. message_count, duration_ms).
        """

    @abstractmethod
    async def recall(
        self, conversation_id: str, query: str, top_k: int = 20
    ) -> list[RecallResult]:
        """Retrieve relevant memory fragments for a query.

        Args:
            conversation_id: Scope recall to this conversation's namespace.
            query: The question or search text.
            top_k: Maximum number of results to return.

        Returns:
            List of RecallResult objects, ranked by relevance.
        """

    @abstractmethod
    async def reset(self) -> None:
        """Reset all state for a fresh benchmark run.

        Must be called between benchmark runs to ensure isolation.
        """

    @abstractmethod
    async def stats(self) -> dict:
        """Return memory system statistics.

        Returns:
            Dict with system-specific stats (e.g. memory_count, index_size).
        """
