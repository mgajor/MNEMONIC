"""Scoring engine — computes accuracy metrics from raw question results."""

from __future__ import annotations

from collections import defaultdict

from rich.console import Console
from rich.table import Table

from mnemonic.types import BenchmarkResult, CategoryResult, QuestionResult

console = Console()


class Scorer:
    """Stateless scorer that takes raw results and produces a BenchmarkResult."""

    @staticmethod
    def compute(
        adapter_name: str,
        llm_name: str,
        dataset_name: str,
        question_results: list[QuestionResult],
        ingest_total_ms: float,
    ) -> BenchmarkResult:
        total = len(question_results)
        correct = sum(1 for qr in question_results if qr.is_correct)

        # Per-category breakdown
        by_cat: dict[str, list[QuestionResult]] = defaultdict(list)
        for qr in question_results:
            by_cat[qr.question_type].append(qr)

        per_category: dict[str, CategoryResult] = {}
        for cat, results in sorted(by_cat.items()):
            cat_correct = sum(1 for qr in results if qr.is_correct)
            per_category[cat] = CategoryResult(
                category=cat,
                total=len(results),
                correct=cat_correct,
            )

        # Latency averages
        avg_recall = (
            sum(qr.recall_ms for qr in question_results) / total if total else 0.0
        )
        avg_answer = (
            sum(qr.answer_ms for qr in question_results) / total if total else 0.0
        )

        return BenchmarkResult(
            adapter_name=adapter_name,
            llm_name=llm_name,
            dataset_name=dataset_name,
            question_results=question_results,
            per_category=per_category,
            total=total,
            correct=correct,
            ingest_total_ms=ingest_total_ms,
            avg_recall_ms=avg_recall,
            avg_answer_ms=avg_answer,
        )

    @staticmethod
    def print_report(result: BenchmarkResult) -> None:
        """Pretty-print a benchmark result to the console."""
        console.print("\n[bold]Results[/bold]")
        console.print(f"  Adapter: [cyan]{result.adapter_name}[/cyan]")
        console.print(f"  LLM:     [cyan]{result.llm_name}[/cyan]")
        console.print(f"  Dataset: [cyan]{result.dataset_name}[/cyan]")
        console.print(
            f"\n  [bold green]Overall: {result.accuracy:.1f}%[/bold green]"
            f"  ({result.correct}/{result.total})"
        )

        # Category table
        table = Table(title="Per-Category Breakdown", show_lines=False)
        table.add_column("Category", style="cyan")
        table.add_column("Correct", justify="right")
        table.add_column("Total", justify="right")
        table.add_column("Accuracy", justify="right", style="bold")

        for cat in sorted(result.per_category):
            cr = result.per_category[cat]
            table.add_row(
                cr.category,
                str(cr.correct),
                str(cr.total),
                f"{cr.accuracy:.1f}%",
            )

        console.print()
        console.print(table)

        # Latency summary
        console.print(f"\n  Ingest total:   {result.ingest_total_ms:>8.0f} ms")
        console.print(f"  Avg recall:     {result.avg_recall_ms:>8.1f} ms")
        console.print(f"  Avg answer:     {result.avg_answer_ms:>8.1f} ms")
        console.print()
