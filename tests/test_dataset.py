"""Tests for dataset loading — JSON, JSONL, and datetime extraction."""

import json
import tempfile
from pathlib import Path

from mnemonic.suites.base import load_locomo_dataset


def _make_dataset_item(
    conv_id: str = "conv-1",
    q_index: int = 0,
    question_type: str = "single_hop",
    sessions: list | None = None,
    datetimes: list | None = None,
) -> dict:
    """Helper to create a minimal LoCoMo dataset item."""
    if sessions is None:
        sessions = [[
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How are you?"},
        ]]
    return {
        "question_id": f"{conv_id}_q{q_index}",
        "question": f"Test question {q_index}?",
        "question_type": question_type,
        "choices": ["A) Yes", "B) No", "C) Maybe"],
        "correct_choice_index": 0,
        "haystack_sessions": sessions,
        "haystack_session_datetimes": datetimes or [],
    }


def test_load_json_format() -> None:
    items = [_make_dataset_item("conv-1", 0), _make_dataset_item("conv-1", 1)]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(items, f)
        path = Path(f.name)

    ds = load_locomo_dataset(path)
    assert ds.name == path.stem
    assert len(ds.questions) == 2
    assert "conv-1" in ds.conversations
    assert len(ds.conversations["conv-1"].messages) == 2
    path.unlink()


def test_load_jsonl_format() -> None:
    items = [_make_dataset_item("conv-1", 0), _make_dataset_item("conv-2", 0)]

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
        path = Path(f.name)

    ds = load_locomo_dataset(path)
    assert len(ds.questions) == 2
    assert len(ds.conversations) == 2
    assert "conv-1" in ds.conversations
    assert "conv-2" in ds.conversations
    path.unlink()


def test_datetime_extraction() -> None:
    items = [_make_dataset_item(
        "conv-1", 0,
        sessions=[[
            {"role": "user", "content": "I got a new job"},
            {"role": "assistant", "content": "Congrats!"},
        ]],
        datetimes=["2024-03-15T10:30:00Z"],
    )]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(items, f)
        path = Path(f.name)

    ds = load_locomo_dataset(path)
    conv = ds.conversations["conv-1"]
    # Messages should have datetime prepended
    assert "[2024-03-15T10:30:00Z]" in conv.messages[0].content
    assert "I got a new job" in conv.messages[0].content
    assert conv.timestamps == ["2024-03-15T10:30:00Z"]
    path.unlink()


def test_multiple_sessions_with_datetimes() -> None:
    items = [_make_dataset_item(
        "conv-1", 0,
        sessions=[
            [{"role": "user", "content": "Session 1 msg"}],
            [{"role": "user", "content": "Session 2 msg"}],
        ],
        datetimes=["2024-01-01T00:00:00Z", "2024-06-15T12:00:00Z"],
    )]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(items, f)
        path = Path(f.name)

    ds = load_locomo_dataset(path)
    conv = ds.conversations["conv-1"]
    assert len(conv.messages) == 2
    assert "[2024-01-01T00:00:00Z]" in conv.messages[0].content
    assert "[2024-06-15T12:00:00Z]" in conv.messages[1].content
    path.unlink()


def test_conversation_deduplication() -> None:
    """Multiple questions from same conversation shouldn't duplicate the conversation."""
    items = [
        _make_dataset_item("conv-1", 0),
        _make_dataset_item("conv-1", 1),
        _make_dataset_item("conv-1", 2),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(items, f)
        path = Path(f.name)

    ds = load_locomo_dataset(path)
    assert len(ds.questions) == 3
    assert len(ds.conversations) == 1
    path.unlink()


def test_dataset_categories() -> None:
    items = [
        _make_dataset_item("conv-1", 0, question_type="single_hop"),
        _make_dataset_item("conv-1", 1, question_type="multi_hop"),
        _make_dataset_item("conv-1", 2, question_type="temporal"),
    ]

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(items, f)
        path = Path(f.name)

    ds = load_locomo_dataset(path)
    assert ds.categories == {"single_hop", "multi_hop", "temporal"}
    path.unlink()
