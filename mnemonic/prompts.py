"""Prompt templates for benchmark answer generation.

v1: Basic numbered context — matches engram-locomo --prompt-version v1
v2: Enriched metadata — dates, epistemic certainty tags, corroboration.
    Matches engram-locomo --prompt-version v2
"""

from __future__ import annotations

from mnemonic.types import RecallResult


# ── v1: Basic prompt ────────────────────────────────────────────

V1_TEMPLATE = """\
Context from a conversation (timestamps in brackets, "yesterday" = day before that timestamp):
{context}

Question: {question}

{choices}

If the answer cannot be determined from the context above, select the choice that says it is not answerable.
Respond with ONLY the letter of the correct answer. Do not explain.
Answer:"""


def format_context_v1(memories: list[RecallResult]) -> str:
    """v1: Simple numbered list of memory contents."""
    return "\n".join(f"[{i + 1}] {m.content}" for i, m in enumerate(memories))


# ── v2: Enriched metadata prompt ───────────────────────────────

V2_TEMPLATE = """\
Context from a conversation. Each memory includes metadata:
- Dates in [brackets] indicate when events occurred; "yesterday" = day before that date
- Certainty tags: [certain], [likely], [vague] indicate reliability
- Source tags: [direct], [consolidated], [inferred] indicate how the memory was formed

{context}

Question: {question}

{choices}

If the answer cannot be determined from the context above, select the choice that says it is not answerable.
Respond with ONLY the letter of the correct answer. Do not explain.
Answer:"""


def format_context_v2(memories: list[RecallResult]) -> str:
    """v2: Enriched context with certainty, kind, and source metadata."""
    lines: list[str] = []
    for i, m in enumerate(memories):
        tags: list[str] = []
        if m.certainty:
            tags.append(m.certainty)
        if m.source:
            tags.append(m.source.lower())
        if m.kind:
            tags.append(m.kind)

        tag_str = f" [{', '.join(tags)}]" if tags else ""
        lines.append(f"[{i + 1}]{tag_str} {m.content}")

    return "\n".join(lines)


# ── Public interface ────────────────────────────────────────────


PROMPT_VERSIONS = {"v1", "v2"}


def _format_choices(choices: list[str]) -> str:
    """Format choices with letter prefixes (A, B, C, ...).

    If choices already have letter prefixes like "A) ...", they're used as-is.
    Otherwise, letter prefixes are added automatically.
    """
    import re

    # Check if first choice already has a letter prefix
    if choices and re.match(r"^[A-Za-z][).]\s", choices[0]):
        return "\n".join(choices)

    return "\n".join(
        f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)
    )


def build_prompt(
    memories: list[RecallResult],
    question: str,
    choices: list[str],
    prompt_version: str = "v1",
) -> str:
    """Build the full answer-generation prompt for the given version.

    Args:
        memories: Recalled memory fragments from the adapter.
        question: The benchmark question text.
        choices: Answer options (with or without letter prefixes).
        prompt_version: "v1" (basic) or "v2" (enriched metadata).

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    choices_str = _format_choices(choices)

    if prompt_version == "v2":
        context = format_context_v2(memories)
        return V2_TEMPLATE.format(
            context=context, question=question, choices=choices_str
        )

    # Default: v1
    context = format_context_v1(memories)
    return V1_TEMPLATE.format(
        context=context, question=question, choices=choices_str
    )
