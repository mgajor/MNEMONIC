from mnemonic.types import (
    BenchmarkResult,
    CategoryResult,
    Dataset,
    Conversation,
    Message,
    Question,
    QuestionResult,
)


def test_question_correct_letter() -> None:
    q = Question(
        question_id="test_q0",
        question="Where does Alice work?",
        question_type="single_hop",
        choices=["A) Acme", "B) Globex"],
        correct_choice_index=1,
        conversation_id="test",
    )
    assert q.correct_letter == "B"


def test_category_result_accuracy() -> None:
    cr = CategoryResult(category="single_hop", total=10, correct=8)
    assert cr.accuracy == 80.0


def test_category_result_empty() -> None:
    cr = CategoryResult(category="empty", total=0, correct=0)
    assert cr.accuracy == 0.0


def test_dataset_categories() -> None:
    ds = Dataset(
        name="test",
        conversations={},
        questions=[
            Question("q1", "?", "single_hop", [], 0, "c1"),
            Question("q2", "?", "multi_hop", [], 0, "c1"),
            Question("q3", "?", "single_hop", [], 0, "c1"),
        ],
    )
    assert ds.categories == {"single_hop", "multi_hop"}


def test_benchmark_result_accuracy() -> None:
    result = BenchmarkResult(
        adapter_name="test",
        llm_name="test",
        dataset_name="test",
        question_results=[],
        per_category={},
        total=20,
        correct=17,
        ingest_total_ms=100.0,
        avg_recall_ms=5.0,
        avg_answer_ms=50.0,
    )
    assert result.accuracy == 85.0
