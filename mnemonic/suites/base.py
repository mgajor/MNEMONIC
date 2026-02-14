"""Base benchmark suite — orchestrates ingest, recall, answer, and scoring."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from mnemonic.adapters.base import BaseAdapter
from mnemonic.llm.base import BaseLLMProvider
from mnemonic.prompts import build_prompt
from mnemonic.scoring.scorer import Scorer
from mnemonic.types import (
    BenchmarkResult,
    Conversation,
    Dataset,
    Message,
    Question,
    QuestionResult,
)

console = Console()


# ── Dataset loading ─────────────────────────────────────────────


def load_locomo_dataset(path: Path) -> Dataset:
    """Load a LoCoMo-MC10 formatted dataset (JSON array or JSONL).

    Supports both formats:
    - JSON: a single array of question objects
    - JSONL: one JSON object per line

    Each object has: question_id, question, question_type, choices,
    correct_choice_index, haystack_sessions, haystack_session_datetimes.
    """
    text = path.read_text()

    # Detect format: JSON array starts with '[', JSONL starts with '{'
    stripped = text.strip()
    if stripped.startswith("["):
        raw = json.loads(text)
    else:
        # JSONL: one JSON object per line
        raw = [json.loads(line) for line in stripped.splitlines() if line.strip()]

    conversations: dict[str, Conversation] = {}
    questions: list[Question] = []

    for item in raw:
        qid = item["question_id"]
        # conversation_id is the prefix before the question number
        conv_id = qid.rsplit("_", 1)[0]

        # Build conversation if we haven't seen this one yet
        if conv_id not in conversations:
            messages: list[Message] = []
            session_datetimes = item.get("haystack_session_datetimes", [])

            for session_idx, session in enumerate(item.get("haystack_sessions", [])):
                # Get the datetime for this session if available
                session_dt = ""
                if session_idx < len(session_datetimes):
                    session_dt = session_datetimes[session_idx]

                for msg in session:
                    content = msg["content"]
                    # Prepend session datetime as context if available
                    if session_dt:
                        content = f"[{session_dt}] {content}"
                    messages.append(Message(role=msg["role"], content=content))

            conversations[conv_id] = Conversation(
                conversation_id=conv_id,
                messages=messages,
                timestamps=session_datetimes if isinstance(session_datetimes, list) else [],
            )

        questions.append(
            Question(
                question_id=qid,
                question=item["question"],
                question_type=item["question_type"],
                choices=item["choices"],
                correct_choice_index=item["correct_choice_index"],
                conversation_id=conv_id,
            )
        )

    return Dataset(
        name=path.stem,
        conversations=conversations,
        questions=questions,
    )


# ── Suite runner ────────────────────────────────────────────────


class BaseSuite:
    """Orchestrates a full benchmark run.

    Takes an adapter (memory system) and an llm_provider (answer generation)
    as independent pluggable components. This separation is intentional —
    the same retrieval quality produces wildly different scores depending
    on which LLM interprets the context.
    """

    # Category-specific top_k multipliers (matches engram-locomo v2).
    # temporal_reasoning needs more context for date ordering;
    # multi_hop needs broader coverage; adversarial benefits from less noise.
    TOP_K_MULTIPLIERS: dict[str, float] = {
        "temporal_reasoning": 1.5,
        "multi_hop": 1.25,
        "adversarial": 0.75,
    }

    # Category → query intent hint for engram recall_with_intent.
    INTENT_MAP: dict[str, str] = {
        "temporal_reasoning": "temporal",
        "single_hop": "factual",
        "open_domain": "general",
        "adversarial": "factual",
        # multi_hop: omitted → auto-detect (heuristic outperforms blanket hint)
    }

    def __init__(
        self,
        adapter: BaseAdapter,
        llm_provider: BaseLLMProvider,
        prompt_version: str = "v1",
        concurrency: int = 1,
        top_k: int = 20,
        use_intent: bool = False,
    ):
        self.adapter = adapter
        self.llm = llm_provider
        self.prompt_version = prompt_version
        self.concurrency = max(1, concurrency)
        self.top_k = top_k
        self.use_intent = use_intent

    def _effective_top_k(self, question_type: str) -> int:
        """Compute dynamic top_k based on question category."""
        multiplier = self.TOP_K_MULTIPLIERS.get(question_type, 1.0)
        return max(5, round(self.top_k * multiplier))

    async def _answer_question(
        self, q: Question, sem: asyncio.Semaphore,
    ) -> QuestionResult:
        """Answer a single question (with semaphore for concurrency control)."""
        async with sem:
            # Recall with category-aware top_k and optional intent hint
            effective_k = self._effective_top_k(q.question_type)
            intent = self.INTENT_MAP.get(q.question_type) if self.use_intent else None
            recall_start = time.perf_counter()
            memories = await self.adapter.recall(
                q.conversation_id, q.question, top_k=effective_k,
                intent=intent,
            )
            recall_ms = (time.perf_counter() - recall_start) * 1000

            # Build prompt using versioned template
            prompt = build_prompt(
                memories=memories,
                question=q.question,
                choices=q.choices,
                prompt_version=self.prompt_version,
            )

            answer_start = time.perf_counter()
            predicted = await self.llm.generate_answer(
                context=prompt, question=q.question, choices=q.choices
            )
            answer_ms = (time.perf_counter() - answer_start) * 1000

            # Normalize predicted answer to single letter
            predicted_letter = predicted.strip().upper()[:1]
            correct_letter = q.correct_letter

            return QuestionResult(
                question_id=q.question_id,
                question_type=q.question_type,
                predicted=predicted_letter,
                correct=correct_letter,
                is_correct=(predicted_letter == correct_letter),
                recall_ms=recall_ms,
                answer_ms=answer_ms,
            )

    async def run(
        self, dataset: Dataset, max_questions: int | None = None
    ) -> BenchmarkResult:
        """Execute the full benchmark: reset -> ingest -> recall+answer -> score.

        Args:
            dataset: The loaded benchmark dataset.
            max_questions: If set, only evaluate this many questions (for quick validation).
        """
        questions = dataset.questions
        if max_questions is not None and max_questions > 0:
            questions = questions[:max_questions]

        # When limiting questions, only ingest conversations we actually need
        needed_conv_ids = {q.conversation_id for q in questions}
        conversations = {
            cid: conv
            for cid, conv in dataset.conversations.items()
            if cid in needed_conv_ids
        }

        console.print(
            f"\n[bold]MNEMONIC Benchmark[/bold]"
            f"\n  Adapter:  [cyan]{self.adapter.name}[/cyan]"
            f"\n  LLM:      [cyan]{self.llm.name}[/cyan]"
            f"\n  Dataset:  [cyan]{dataset.name}[/cyan]"
            f"\n  Prompt:   [cyan]{self.prompt_version}[/cyan]"
            f"\n  Concurrency: [cyan]{self.concurrency}[/cyan]"
            f"\n  Questions: {len(questions)}"
            + (f" (of {len(dataset.questions)})" if max_questions else "")
            + f"\n  Conversations: {len(conversations)}\n"
        )

        # ── Reset ───────────────────────────────────────────────
        await self.adapter.reset()

        # ── Ingest phase ────────────────────────────────────────
        ingest_start = time.perf_counter()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Ingesting conversations...", total=len(conversations)
            )
            for conv in conversations.values():
                await self.adapter.ingest(conv, concurrency=self.concurrency)
                progress.advance(task)

        ingest_total_ms = (time.perf_counter() - ingest_start) * 1000
        console.print(f"  Ingest complete: {ingest_total_ms:.0f}ms total\n")

        # ── Recall + Answer phase ──────────────────────────────
        sem = asyncio.Semaphore(self.concurrency)

        if self.concurrency <= 1:
            # Sequential mode — shows nice progress
            question_results: list[QuestionResult] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                ptask = progress.add_task(
                    "Answering questions...", total=len(questions)
                )
                for q in questions:
                    result = await self._answer_question(q, sem)
                    question_results.append(result)
                    progress.advance(ptask)
        else:
            # Concurrent mode — gather all at once
            console.print(
                f"  Answering {len(questions)} questions "
                f"({self.concurrency} concurrent)..."
            )

            async def _tracked_answer(q: Question) -> QuestionResult:
                result = await self._answer_question(q, sem)
                return result

            question_results = list(await asyncio.gather(
                *(_tracked_answer(q) for q in questions)
            ))
            console.print(f"  Answering complete.\n")

        # ── Score ──────────────────────────────────────────────
        result = Scorer.compute(
            adapter_name=self.adapter.name,
            llm_name=self.llm.name,
            dataset_name=dataset.name,
            question_results=question_results,
            ingest_total_ms=ingest_total_ms,
        )

        return result
