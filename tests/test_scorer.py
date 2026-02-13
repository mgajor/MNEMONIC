from mnemonic.scoring.scorer import Scorer
from mnemonic.types import QuestionResult


def _make_result(qid: str, qtype: str, predicted: str, correct: str) -> QuestionResult:
    return QuestionResult(
        question_id=qid,
        question_type=qtype,
        predicted=predicted,
        correct=correct,
        is_correct=(predicted == correct),
        recall_ms=5.0,
        answer_ms=50.0,
    )


def test_scorer_compute_basic() -> None:
    results = [
        _make_result("q1", "single_hop", "A", "A"),
        _make_result("q2", "single_hop", "B", "A"),
        _make_result("q3", "multi_hop", "C", "C"),
        _make_result("q4", "temporal", "A", "B"),
    ]

    br = Scorer.compute(
        adapter_name="test_adapter",
        llm_name="test_llm",
        dataset_name="test_dataset",
        question_results=results,
        ingest_total_ms=200.0,
    )

    assert br.total == 4
    assert br.correct == 2
    assert br.accuracy == 50.0
    assert br.adapter_name == "test_adapter"
    assert br.llm_name == "test_llm"

    # Per-category
    assert br.per_category["single_hop"].total == 2
    assert br.per_category["single_hop"].correct == 1
    assert br.per_category["multi_hop"].accuracy == 100.0
    assert br.per_category["temporal"].accuracy == 0.0

    # Latency
    assert br.avg_recall_ms == 5.0
    assert br.avg_answer_ms == 50.0
    assert br.ingest_total_ms == 200.0


def test_scorer_empty() -> None:
    br = Scorer.compute(
        adapter_name="empty",
        llm_name="empty",
        dataset_name="empty",
        question_results=[],
        ingest_total_ms=0.0,
    )
    assert br.total == 0
    assert br.accuracy == 0.0
