from __future__ import annotations

import importlib
import json

import gradio as gr
import pytest

import app
from models.analyzer import DemoBackend, SentinexAnalyzer


def test_import_is_server_and_model_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_launch(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"launch called during import: {args!r} {kwargs!r}")

    monkeypatch.setattr(gr.Blocks, "launch", fail_launch)
    importlib.reload(app)

    assert app.get_analyzer.cache_info().currsize == 0


def test_callback_returns_structured_nonclinical_output_without_raw_text() -> None:
    private_text = "<script>alert('private')</script> I feel alone today."

    outputs = app.analyze_text(private_text, [], SentinexAnalyzer(DemoBackend()))
    serialized = json.dumps(outputs)

    assert len(outputs) == 7
    assert "synthetic" in outputs[0]
    assert outputs[3] == [
        {
            "category": "social_disconnection_wording",
            "literal_matches": ["feel alone"],
        }
    ]
    assert private_text not in serialized
    assert "<script>" not in serialized
    assert "risk_level" not in serialized
    assert "diagnosis" not in serialized.casefold()


def test_callback_keeps_sessions_isolated() -> None:
    analyzer = SentinexAnalyzer()

    session_a = app.analyze_text("Session A fixture.", [], analyzer)[-1]
    session_b = app.analyze_text("Session B fixture.", [], analyzer)[-1]
    session_a_next = app.analyze_text("Session A follow-up.", session_a, analyzer)[-1]

    assert len(session_a) == 1
    assert len(session_b) == 1
    assert len(session_a_next) == 2


def test_callback_rejects_invalid_input() -> None:
    with pytest.raises(gr.Error, match="at least 5"):
        app.analyze_text("no", [], SentinexAnalyzer())


def test_app_builds_without_loading_analyzer(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get_analyzer() -> None:
        raise AssertionError("analyzer loaded while constructing UI")

    monkeypatch.setattr(app, "get_analyzer", fail_get_analyzer)

    interface = app.build_app()

    assert isinstance(interface, gr.Blocks)


def test_main_applies_host_theme_and_css(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeApp:
        def launch(self, **kwargs: object) -> None:
            observed.update(kwargs)

    monkeypatch.setenv("SENTINEX_HOST", "127.0.0.9")
    monkeypatch.setattr(app, "build_app", FakeApp)

    app.main()

    assert observed["server_name"] == "127.0.0.9"
    assert observed["server_port"] == 7860
    assert isinstance(observed["theme"], gr.themes.Soft)
    assert observed["css"] == app.APP_CSS


def test_clear_session_removes_aggregate_state() -> None:
    cleared = app.clear_session()

    assert cleared[0] == ""
    assert cleared[-1] == []
    assert cleared[2:6] == ([], [], [], [])
