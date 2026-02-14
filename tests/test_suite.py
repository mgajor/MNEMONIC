"""Tests for BaseSuite — dynamic top_k and category multipliers."""

from mnemonic.suites.base import BaseSuite


class _DummyAdapter:
    name = "dummy"
    version = "0.0"


class _DummyLLM:
    name = "dummy"


def _make_suite(top_k: int = 20) -> BaseSuite:
    # BaseSuite only needs adapter/llm for run(); we just test _effective_top_k
    return BaseSuite(
        adapter=_DummyAdapter(),  # type: ignore[arg-type]
        llm_provider=_DummyLLM(),  # type: ignore[arg-type]
        top_k=top_k,
    )


def test_effective_top_k_temporal() -> None:
    suite = _make_suite(20)
    assert suite._effective_top_k("temporal_reasoning") == 30  # 20 * 1.5


def test_effective_top_k_multi_hop() -> None:
    suite = _make_suite(20)
    assert suite._effective_top_k("multi_hop") == 25  # 20 * 1.25


def test_effective_top_k_adversarial() -> None:
    suite = _make_suite(20)
    assert suite._effective_top_k("adversarial") == 15  # 20 * 0.75


def test_effective_top_k_default() -> None:
    suite = _make_suite(20)
    assert suite._effective_top_k("single_hop") == 20  # 1.0x
    assert suite._effective_top_k("open_domain") == 20


def test_effective_top_k_minimum() -> None:
    suite = _make_suite(4)
    # 4 * 0.75 = 3, but minimum is 5
    assert suite._effective_top_k("adversarial") == 5
