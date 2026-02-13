"""Shared data models for the MNEMONIC benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Dataset models ──────────────────────────────────────────────


@dataclass
class Message:
    role: str  # "user" or "assistant"
    content: str


@dataclass
class Conversation:
    conversation_id: str
    messages: list[Message]
    timestamps: list[str] = field(default_factory=list)


@dataclass
class Question:
    question_id: str
    question: str
    question_type: str  # single_hop, multi_hop, open_domain, adversarial, temporal
    choices: list[str]
    correct_choice_index: int
    conversation_id: str

    @property
    def correct_letter(self) -> str:
        return chr(ord("A") + self.correct_choice_index)


@dataclass
class Dataset:
    name: str
    conversations: dict[str, Conversation]
    questions: list[Question]

    @property
    def categories(self) -> set[str]:
        return {q.question_type for q in self.questions}


# ── Result models ───────────────────────────────────────────────


@dataclass
class QuestionResult:
    question_id: str
    question_type: str
    predicted: str  # answer letter e.g. "A"
    correct: str  # correct letter e.g. "B"
    is_correct: bool
    recall_ms: float
    answer_ms: float


@dataclass
class CategoryResult:
    category: str
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100) if self.total > 0 else 0.0


@dataclass
class BenchmarkResult:
    adapter_name: str
    llm_name: str
    dataset_name: str
    question_results: list[QuestionResult]
    per_category: dict[str, CategoryResult]
    total: int
    correct: int
    ingest_total_ms: float
    avg_recall_ms: float
    avg_answer_ms: float

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total * 100) if self.total > 0 else 0.0
