"""Base benchmark suite — orchestrates ingest, recall, answer, and scoring."""

from __future__ import annotations

import json
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from mnemonic.adapters.base import BaseAdapter
from mnemonic.llm.base import BaseLLMProvider
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


def load_locomo_dataset(path: Path) -> Dataset:
    """Load a LoCoMo-MC10 formatted JSON dataset.

    Expected format: list of objects, each with question_id, question,
    question_type, choices, correct_choice_index, haystack_sessions,
    and haystack_session_datetimes.
    """
    raw = json.loads(path.read_text())

    conversations: dict[str, Conversation] = {}
    questions: list[Question] = []

    for item in raw:
        qid = item["question_id"]
        # conversation_id is the prefix before the question number
        conv_id = qid.rsplit("_", 1)[0]

        # Build conversation if we haven't seen this one yet
        if conv_id not in conversations:
            messages: list[Message] = []
            for session in item.get("haystack_sessions", []):
                for msg in session:
                    messages.append(Message(role=msg["role"], content=msg["content"]))

            timestamps = item.get("haystack_session_datetimes", [])
            conversations[conv_id] = Conversation(
                conversation_id=conv_id,
                messages=messages,
                timestamps=timestamps if isinstance(timestamps, list) else [],
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


# ── Answer prompt template ──────────────────────────────────────

ANSWER_PROMPT_TEMPLATE = """\
Context from a conversation (timestamps in brackets, "yesterday" = day before that timestamp):
{context}

Question: {question}

{choices}

If the answer cannot be determined from the context above, select the choice that says it is not answerable.
Respond with ONLY the letter of the correct answer. Do not explain.
Answer:"""


def format_context(memories: list[str]) -> str:
    return "\n".join(f"[{i + 1}] {m}" for i, m in enumerate(memories))


def format_choices(choices: list[str]) -> str:
    return "\n".join(choices)


# ── Suite runner ────────────────────────────────────────────────


class BaseSuite:
    """Orchestrates a full benchmark run.

    Takes an adapter (memory system) and an llm_provider (answer generation)
    as independent pluggable components. This separation is intentional —
    the same retrieval quality produces wildly different scores depending
    on which LLM interprets the context.
    """

    def __init__(self, adapter: BaseAdapter, llm_provider: BaseLLMProvider):
        self.adapter = adapter
        self.llm = llm_provider

    async def run(self, dataset: Dataset) -> BenchmarkResult:
        """Execute the full benchmark: reset -> ingest -> recall+answer -> score."""

        console.print(
            f"\n[bold]MNEMONIC Benchmark[/bold]"
            f"\n  Adapter: [cyan]{self.adapter.name}[/cyan]"
            f"\n  LLM:     [cyan]{self.llm.name}[/cyan]"
            f"\n  Dataset: [cyan]{dataset.name}[/cyan]"
            f"\n  Questions: {len(dataset.questions)}"
            f"\n  Conversations: {len(dataset.conversations)}\n"
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
                "Ingesting conversations...", total=len(dataset.conversations)
            )
            for conv in dataset.conversations.values():
                await self.adapter.ingest(conv)
                progress.advance(task)

        ingest_total_ms = (time.perf_counter() - ingest_start) * 1000
        console.print(f"  Ingest complete: {ingest_total_ms:.0f}ms total\n")

        # ── Recall + Answer phase ──────────────────────────────
        question_results: list[QuestionResult] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Answering questions...", total=len(dataset.questions)
            )

            for q in dataset.questions:
                # Recall
                recall_start = time.perf_counter()
                memories = await self.adapter.recall(q.conversation_id, q.question)
                recall_ms = (time.perf_counter() - recall_start) * 1000

                # Build prompt and generate answer
                context = format_context(memories)
                choices_str = format_choices(q.choices)
                prompt = ANSWER_PROMPT_TEMPLATE.format(
                    context=context, question=q.question, choices=choices_str
                )

                answer_start = time.perf_counter()
                predicted = await self.llm.generate_answer(
                    context=prompt, question=q.question, choices=q.choices
                )
                answer_ms = (time.perf_counter() - answer_start) * 1000

                # Normalize predicted answer to single letter
                predicted_letter = predicted.strip().upper()[:1]
                correct_letter = q.correct_letter

                question_results.append(
                    QuestionResult(
                        question_id=q.question_id,
                        question_type=q.question_type,
                        predicted=predicted_letter,
                        correct=correct_letter,
                        is_correct=(predicted_letter == correct_letter),
                        recall_ms=recall_ms,
                        answer_ms=answer_ms,
                    )
                )

                progress.advance(task)

        # ── Score ──────────────────────────────────────────────
        result = Scorer.compute(
            adapter_name=self.adapter.name,
            llm_name=self.llm.name,
            dataset_name=dataset.name,
            question_results=question_results,
            ingest_total_ms=ingest_total_ms,
        )

        return result
