"""Engram memory system adapter — CLI subprocess integration."""

from __future__ import annotations

import asyncio
import logging
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from mnemonic.adapters.base import BaseAdapter
from mnemonic.types import Conversation

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

# Matches lines like:
# 1. [0.87] (certain) (Semantic) Alice works at Acme Corp (id: a1b2c3d4, source: Direct)
_RECALL_LINE_RE = re.compile(
    r"^\d+\.\s+"  # index
    r"\[[\d.]+\]\s+"  # score
    r"\([^)]+\)\s+"  # certainty
    r"\([^)]+\)\s+"  # kind
    r"(.+?)\s+"  # content (captured)
    r"\(id:\s*\w+,\s*source:\s*[^)]+\)$"  # id + source trailer
)


def parse_recall_output(stdout: str) -> list[str]:
    """Parse engram recall CLI output into a list of memory content strings."""
    results: list[str] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue

        match = _RECALL_LINE_RE.match(line)
        if match:
            results.append(match.group(1))
        else:
            # Fallback: if the format doesn't match the regex exactly,
            # try to extract content between the kind tag and the id trailer.
            # This handles cases where content itself contains parentheses.
            fallback = _parse_recall_line_fallback(line)
            if fallback:
                results.append(fallback)

    return results


def _parse_recall_line_fallback(line: str) -> str | None:
    """Fallback parser for recall lines that don't match the strict regex.

    Strategy: find the third ')' (after score, certainty, kind) and the
    last '(id:' marker, and extract everything in between.
    """
    # Find "(id:" marker near the end
    id_marker = line.rfind("(id:")
    if id_marker == -1:
        return None

    # Find the content start: after the second closing paren
    # Format: {index}. [{score}] ({certainty}) ({kind}) {content} (id: ...)
    # The brackets [] don't count — only the two (...) groups before content.
    paren_count = 0
    content_start = -1
    for i, ch in enumerate(line):
        if ch == ")":
            paren_count += 1
            if paren_count == 2:
                content_start = i + 1
                break

    if content_start == -1 or content_start >= id_marker:
        return None

    content = line[content_start:id_marker].strip()
    return content if content else None


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

    async def ingest(self, conversation: Conversation) -> dict:
        """Ingest a conversation by observing each message in its namespace."""
        namespace = conversation.conversation_id
        self._namespaces.add(namespace)

        start = time.perf_counter()
        ingested = 0

        for msg in conversation.messages:
            role = msg.role if msg.role in ("user", "assistant") else "observation"
            _, _, rc = await self._run(
                namespace, "observe", msg.content, "--role", role
            )
            if rc == 0:
                ingested += 1

        duration_ms = (time.perf_counter() - start) * 1000

        return {
            "conversation_id": namespace,
            "message_count": len(conversation.messages),
            "ingested": ingested,
            "duration_ms": duration_ms,
        }

    async def recall(
        self, conversation_id: str, query: str, top_k: int = 20
    ) -> list[str]:
        """Retrieve relevant memories from engram via semantic recall."""
        stdout, _, _ = await self._run(
            conversation_id, "recall", query, "--limit", str(top_k)
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
