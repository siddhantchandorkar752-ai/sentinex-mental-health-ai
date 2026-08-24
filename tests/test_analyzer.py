from __future__ import annotations

import json

import pytest

from models.analyzer import (
    EMOTION_LABELS,
    EMOTION_MODEL,
    EMOTION_REVISION,
    MAX_HISTORY_ENTRIES,
    MAX_INPUT_CHARACTERS,
    SENTIMENT_LABELS,
    SENTIMENT_MODEL,
    SENTIMENT_REVISION,
    DemoBackend,
    ExecutionMode,
    SentinexAnalyzer,
    TransformersBackend,
    build_analyzer,
)


def test_demo_backend_is_deterministic_input_dependent_and_offline() -> None:
    analyzer = SentinexAnalyzer(DemoBackend())

    first = analyzer.analyze("A stable offline fixture.")
    repeated = analyzer.analyze("A stable offline fixture.")
    changed = analyzer.analyze("A different offline fixture.")

    assert first.emotions == repeated.emotions
    assert first.sentiments == repeated.sentiments
    assert first.emotions != changed.emotions
    assert first.evidence_status == "synthetic"
    assert first.backend_details["network"] == "disabled"
    assert sum(item.score for item in first.emotions) == pytest.approx(1.0, abs=1e-5)


def test_default_build_uses_demo_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SENTINEX_MODE", raising=False)

    result = build_analyzer().analyze("Default mode stays offline.")

    assert result.execution_mode is ExecutionMode.DEMO


@pytest.mark.parametrize("text", ["", "   ", "four"])
def test_short_input_is_rejected(text: str) -> None:
    with pytest.raises(ValueError, match="at least 5"):
        SentinexAnalyzer().analyze(text)


def test_oversized_input_is_rejected() -> None:
    with pytest.raises(ValueError, match=str(MAX_INPUT_CHARACTERS)):
        SentinexAnalyzer().analyze("x" * (MAX_INPUT_CHARACTERS + 1))


def test_history_is_caller_owned_bounded_and_contains_no_raw_text() -> None:
    analyzer = SentinexAnalyzer()
    private_text = "private phrase that must never be retained"

    first = analyzer.analyze(private_text)
    second = analyzer.analyze("A separate session value.")
    history = first.history
    for index in range(MAX_HISTORY_ENTRIES + 4):
        history = analyzer.analyze(f"History fixture number {index}.", history).history

    assert len(first.history) == 1
    assert len(second.history) == 1
    assert len(history) == MAX_HISTORY_ENTRIES
    assert private_text not in json.dumps([item.to_mapping() for item in history])


def test_literal_matches_use_phrase_boundaries() -> None:
    analyzer = SentinexAnalyzer()

    matched = analyzer.analyze("I feel alone today.")
    not_matched = analyzer.analyze("The profile_aloneish token is synthetic.")

    assert [item.category for item in matched.lexical_signals] == ["social_disconnection_wording"]
    assert not not_matched.lexical_signals


def test_literal_matching_discloses_context_limitations() -> None:
    result = SentinexAnalyzer().analyze('The quoted sentence says "I want to die".')

    assert result.lexical_signals
    assert any("quotation" in limitation for limitation in result.limitations)
    assert any("intent" in limitation for limitation in result.limitations)


class FakePipeline:
    def __init__(self, labels: tuple[str, ...]) -> None:
        self.labels = labels

    def __call__(self, text: str, **kwargs: object) -> list[dict[str, str | float]]:
        assert text
        assert kwargs == {"top_k": None, "truncation": True, "max_length": 512}
        score = 1.0 / len(self.labels)
        return [{"label": label, "score": score} for label in self.labels]


def test_transformer_adapter_pins_revisions_and_disables_remote_code() -> None:
    calls: list[dict[str, object]] = []

    def factory(task: str, **kwargs: object) -> FakePipeline:
        calls.append({"task": task, **kwargs})
        labels = EMOTION_LABELS if kwargs["model"] == EMOTION_MODEL else SENTIMENT_LABELS
        return FakePipeline(labels)

    result = SentinexAnalyzer(TransformersBackend(factory)).analyze("Adapter contract fixture.")

    assert result.execution_mode is ExecutionMode.TRANSFORMERS
    assert calls == [
        {
            "task": "text-classification",
            "model": EMOTION_MODEL,
            "revision": EMOTION_REVISION,
            "trust_remote_code": False,
        },
        {
            "task": "text-classification",
            "model": SENTIMENT_MODEL,
            "revision": SENTIMENT_REVISION,
            "trust_remote_code": False,
        },
    ]


def test_transformer_adapter_rejects_ambiguous_labels() -> None:
    def factory(task: str, **kwargs: object) -> FakePipeline:
        del task, kwargs
        return FakePipeline(("LABEL_0", "LABEL_1"))

    with pytest.raises(ValueError, match="unsupported label semantics"):
        SentinexAnalyzer(TransformersBackend(factory)).analyze("Fail closed fixture.")
