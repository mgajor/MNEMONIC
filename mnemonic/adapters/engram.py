"""Engram memory system adapter — CLI subprocess integration."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from mnemonic.adapters.base import BaseAdapter
from mnemonic.types import Conversation, RecallResult

logger = logging.getLogger(__name__)


@dataclass
class EngramConfig:
    """Configuration for the Engram adapter."""

    binary_path: str = "engram-mcp"
    db_path: str = ""  # empty = auto-create temp file per run
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "mxbai-embed-large"
    top_k: int = 20

    def __post_init__(self) -> None:
        if not self.db_path:
            self.db_path = tempfile.mktemp(suffix=".db", prefix="mnemonic_engram_")


# ── Recall output parser ───────────────────────────────────────

# Matches the actual engram-mcp recall output format:
# [0.70|certain|Direct|Semantic] Alice works at Acme Corp
_RECALL_LINE_RE = re.compile(
    r"^\[([^|]+)\|([^|]+)\|([^|]+)\|([^]]+)\]\s+(.+)$"
)


def parse_recall_output(stdout: str) -> list[RecallResult]:
    """Parse engram recall CLI output into a list of RecallResult objects.

    Expected format per line:
        [score|certainty|source|kind] content text here
    """
    results: list[RecallResult] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = _RECALL_LINE_RE.match(line)
        if match:
            results.append(RecallResult(
                content=match.group(5),
                score=float(match.group(1)),
                certainty=match.group(2).lower(),
                kind=match.group(4).lower(),
                source=match.group(3),
            ))

    return results


# ── Adapter ────────────────────────────────────────────────────


class EngramAdapter(BaseAdapter):
    """MNEMONIC adapter for the Engram memory system.

    Uses the engram-mcp CLI binary via async subprocess calls.
    Each conversation is isolated in its own namespace.
    """

    name = "engram"
    version = "0.1.0"

    def __init__(self, config: EngramConfig | None = None) -> None:
        self.config = config or EngramConfig()
        self._namespaces: set[str] = set()

    def _base_args(self, namespace: str) -> list[str]:
        """Build the common CLI arguments."""
        return [
            self.config.binary_path,
            "--db-path", self.config.db_path,
            "--namespace", namespace,
            "--ollama-url", self.config.ollama_url,
            "--embed-model", self.config.embed_model,
        ]

    async def _run(
        self, namespace: str, subcommand: str, *args: str
    ) -> tuple[str, str, int]:
        """Run an engram-mcp CLI subcommand asynchronously.

        Returns:
            Tuple of (stdout, stderr, returncode).
        """
        cmd = [*self._base_args(namespace), subcommand, *args]
        logger.debug("engram cmd: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
        stdout = stdout_bytes.decode()
        stderr = stderr_bytes.decode()

        if proc.returncode != 0:
            logger.warning(
                "engram %s returned %d: %s", subcommand, proc.returncode, stderr
            )

        return stdout, stderr, proc.returncode

    async def ingest(
        self, conversation: Conversation, concurrency: int = 1
    ) -> dict:
        """Ingest a conversation via batch-observe (single engine session).

        Writes all messages to a temp JSONL file, then calls
        ``engram-mcp batch-observe <file> --lifecycle`` once. This avoids
        cold-booting the engine per message.

        Args:
            conversation: The conversation to ingest.
            concurrency: Ignored (kept for API compatibility).
        """
        namespace = conversation.conversation_id
        self._namespaces.add(namespace)

        start = time.perf_counter()

        # Write JSONL temp file
        jsonl_path = Path(tempfile.mktemp(suffix=".jsonl", prefix="engram_batch_"))
        try:
            with open(jsonl_path, "w") as f:
                for msg in conversation.messages:
                    role = msg.role if msg.role in ("user", "assistant") else "observation"
                    line = json.dumps({"content": msg.content, "role": role})
                    f.write(line + "\n")

            stdout, stderr, rc = await self._run(
                namespace, "batch-observe", str(jsonl_path),
            )

            # Parse the JSON summary from stdout
            ingested = 0
            if rc == 0 and stdout.strip():
                try:
                    summary = json.loads(stdout)
                    ingested = summary.get("memories_created", 0)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse batch-observe summary: %s", stdout[:200])
        finally:
            jsonl_path.unlink(missing_ok=True)

        duration_ms = (time.perf_counter() - start) * 1000

        return {
            "conversation_id": namespace,
            "message_count": len(conversation.messages),
            "ingested": ingested,
            "duration_ms": duration_ms,
        }

    async def recall(
        self, conversation_id: str, query: str, top_k: int = 20
    ) -> list[RecallResult]:
        """Retrieve relevant memories from engram via semantic recall."""
        stdout, _, _ = await self._run(
            conversation_id, "recall", query,
            "--limit", str(top_k),
            "--min-age-secs", "0",
        )
        return parse_recall_output(stdout)

    async def reset(self) -> None:
        """Reset by switching to a fresh temp database."""
        # Remove the old DB file if it exists and was auto-created
        old_path = Path(self.config.db_path)
        if old_path.exists():
            old_path.unlink(missing_ok=True)
            # Also remove WAL/SHM files if present (SQLite)
            old_path.with_suffix(".db-wal").unlink(missing_ok=True)
            old_path.with_suffix(".db-shm").unlink(missing_ok=True)

        # Create a new temp path for the next run
        self.config.db_path = tempfile.mktemp(
            suffix=".db", prefix="mnemonic_engram_"
        )
        self._namespaces.clear()

    async def stats(self) -> dict:
        """Return basic stats about the current engram state."""
        db_path = Path(self.config.db_path)
        return {
            "db_path": self.config.db_path,
            "db_exists": db_path.exists(),
            "db_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
            "namespaces_used": len(self._namespaces),
            "namespace_ids": sorted(self._namespaces),
        }
