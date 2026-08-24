"""Gradio interface for the SENTINEX language-signal research workbench."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

import gradio as gr

from models.analyzer import (
    LIMITATIONS,
    MAX_INPUT_CHARACTERS,
    SAFETY_NOTICE,
    AnalysisResult,
    SentinexAnalyzer,
    build_analyzer,
)

APP_CSS = """
html, body, gradio-app {
    max-width: 100vw !important;
    overflow-x: hidden !important;
}
.gradio-container {
    width: 100% !important;
    min-width: 0 !important;
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 28px 22px 48px !important;
    background:
        radial-gradient(circle at 8% 0%, rgba(20, 184, 166, 0.12), transparent 32rem),
        radial-gradient(circle at 96% 18%, rgba(14, 165, 233, 0.10), transparent 28rem);
}
.sentinex-hero {
    padding: 28px 30px 26px;
    border: 1px solid rgba(45, 212, 191, 0.28);
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(15, 118, 110, 0.15), rgba(2, 132, 199, 0.08));
    box-shadow: 0 18px 60px rgba(2, 8, 23, 0.16);
}
.sentinex-hero h1 {
    margin: 0 0 6px !important;
    letter-spacing: -0.045em;
    font-size: clamp(2.5rem, 7vw, 4.75rem) !important;
    line-height: 0.95 !important;
}
.sentinex-hero h3 {
    margin: 12px 0 16px !important;
    color: var(--body-text-color-subdued);
}
.sentinex-eyebrow code {
    display: inline-block;
    margin: 0 6px 6px 0;
    padding: 5px 10px;
    border: 1px solid rgba(45, 212, 191, 0.34);
    border-radius: 999px;
    background: rgba(13, 148, 136, 0.10);
    color: rgb(45, 212, 191);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
}
.sentinex-safety {
    margin: 14px 0 18px;
    padding: 4px 18px;
    border-left: 4px solid rgb(245, 158, 11);
    border-radius: 12px;
    background: rgba(245, 158, 11, 0.08);
}
.sentinex-workspace {
    gap: 18px !important;
    align-items: stretch !important;
}
.sentinex-panel {
    padding: 14px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: var(--background-fill-secondary);
    box-shadow: 0 10px 35px rgba(2, 8, 23, 0.10);
}
.sentinex-primary button {
    min-height: 48px;
    font-weight: 700;
    letter-spacing: -0.01em;
}
.sentinex-tabs {
    margin-top: 18px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    overflow: hidden;
    background: var(--background-fill-secondary);
}
.sentinex-limitations {
    margin-top: 18px;
    padding: 18px 22px;
    border-radius: 16px;
    background: rgba(100, 116, 139, 0.08);
}
@media (max-width: 700px) {
    .gradio-container {
        width: 100vw !important;
        max-width: 100vw !important;
        min-width: 0 !important;
        padding: 14px 12px 30px !important;
        box-sizing: border-box !important;
    }
    .gradio-container .main {
        width: 100% !important;
        min-width: 0 !important;
        padding: 10px 8px !important;
        box-sizing: border-box !important;
    }
    .gradio-container .contain,
    .gradio-container .wrap,
    .gradio-container .column,
    .gradio-container .row,
    .gradio-container .tabs,
    .gradio-container .tab-wrapper,
    .gradio-container .tab-container,
    .gradio-container .tabitem {
        width: 100% !important;
        min-width: 0 !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }
    .sentinex-workspace {
        flex-direction: column !important;
    }
    .sentinex-workspace > * {
        width: 100% !important;
        min-width: 0 !important;
    }
    .sentinex-hero { padding: 22px 18px; border-radius: 18px; }
    .sentinex-panel { padding: 8px; }
    .tabs > .tab-nav {
        overflow-x: auto !important;
        scrollbar-width: thin;
    }
}
"""


@lru_cache(maxsize=1)
def get_analyzer() -> SentinexAnalyzer:
    """Load the configured backend once, only after the first analysis request."""

    return build_analyzer()


def _history_rows(result: AnalysisResult) -> list[list[str | int | float]]:
    return [
        [
            index,
            entry.top_emotion_label,
            round(entry.top_emotion_score, 4),
            entry.sentiment_label,
            round(entry.sentiment_score, 4),
            entry.lexical_category_count,
            entry.execution_mode,
        ]
        for index, entry in enumerate(result.history, start=1)
    ]


def analyze_text(
    text: str,
    history: Sequence[Mapping[str, Any]] | None,
    analyzer: SentinexAnalyzer | None = None,
) -> tuple[
    str,
    list[list[str | float]],
    list[list[str | float]],
    list[dict[str, str | list[str]]],
    list[list[str | int | float]],
    dict[str, Any],
    list[dict[str, str | int | float]],
]:
    """Analyze text without returning or retaining the submitted text."""

    active_analyzer = analyzer or get_analyzer()
    try:
        result = active_analyzer.analyze(text, history)
    except (TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    mode_note = (
        "Synthetic demo output — useful for testing the interface, not language meaning."
        if result.evidence_status == "synthetic"
        else (
            "Pinned model output — labels are non-clinical and have not been validated "
            "for safety decisions."
        )
    )
    status = (
        f"**Mode:** `{result.execution_mode.value}` · "
        f"**Evidence:** `{result.evidence_status}`  \n{mode_note}"
    )
    emotions = [[item.label, round(item.score, 6)] for item in result.emotions]
    sentiments = [[item.label, round(item.score, 6)] for item in result.sentiments]
    lexical = [item.to_mapping() for item in result.lexical_signals]
    report = {
        "execution_mode": result.execution_mode.value,
        "evidence_status": result.evidence_status,
        "top_emotion_label": result.top_emotion.label,
        "top_sentiment_label": result.top_sentiment.label,
        "literal_match_category_count": len(result.lexical_signals),
        "backend_details": dict(result.backend_details),
        "limitations": list(result.limitations),
    }
    state = [entry.to_mapping() for entry in result.history]
    return (
        status,
        emotions,
        sentiments,
        lexical,
        _history_rows(result),
        report,
        state,
    )


ClearResult = tuple[
    str,
    str,
    list[Any],
    list[Any],
    list[Any],
    list[Any],
    dict[Any, Any],
    list[Any],
]


def clear_session() -> ClearResult:
    """Clear submitted text, user-visible aggregates, and per-session state."""

    return (
        "",
        "Session aggregates cleared.",
        [],
        [],
        [],
        [],
        {},
        [],
    )


def build_app() -> gr.Blocks:
    """Construct the UI without loading or downloading model weights."""

    with gr.Blocks(
        title="SENTINEX · Language-Signal Research Workbench",
        fill_width=True,
    ) as demo:
        gr.Markdown(
            """
            # SENTINEX
            ### A transparent, non-clinical language-signal research workbench

            SENTINEX displays emotion/sentiment classifier labels and literal phrase matches.
            It does **not** diagnose a condition, detect hidden intent, assess crisis risk, or
            determine whether a person is safe.
            """,
            elem_classes=["sentinex-hero"],
        )
        gr.Markdown(
            "`NON-CLINICAL` `SESSION-ISOLATED` `EVIDENCE-LABELED` `OFFLINE BY DEFAULT`",
            elem_classes=["sentinex-eyebrow"],
        )
        gr.Markdown(
            f"**Safety note:** {SAFETY_NOTICE}",
            elem_classes=["sentinex-safety"],
        )

        session_history = gr.State(value=[])
        with gr.Row(elem_classes=["sentinex-workspace"]):
            with gr.Column(scale=5, elem_classes=["sentinex-panel"]):
                text_input = gr.Textbox(
                    label="Text to inspect",
                    placeholder=(
                        "Enter at least five characters. Submitted text is not stored in "
                        "session history."
                    ),
                    lines=7,
                    max_length=MAX_INPUT_CHARACTERS,
                )
                with gr.Row(elem_classes=["sentinex-primary"]):
                    analyze_button = gr.Button(
                        "Inspect language signals",
                        variant="primary",
                    )
                    clear_button = gr.Button("Clear session")
            with gr.Column(scale=4, elem_classes=["sentinex-panel"]):
                status_output = gr.Markdown("No analysis has run in this session.")
                gr.Markdown(
                    "**Interpretation boundary:** a high model score is confidence in a "
                    "model label, "
                    "not severity, probability of a condition, or a safety measurement."
                )

        with gr.Tab("Classifier labels", elem_classes=["sentinex-tabs"]), gr.Row():
            emotion_output = gr.Dataframe(
                headers=["emotion label", "model score"],
                datatype=["str", "number"],
                interactive=False,
                label="Emotion distribution",
            )
            sentiment_output = gr.Dataframe(
                headers=["sentiment label", "model score"],
                datatype=["str", "number"],
                interactive=False,
                label="Sentiment distribution",
            )
        with gr.Tab("Literal matches"):
            gr.Markdown(
                "These are case-insensitive, whole-phrase matches. They do not understand "
                "negation, quotations, context, intent, or sarcasm."
            )
            lexical_output = gr.JSON(label="Matched wording categories")
        with gr.Tab("Session aggregates"):
            gr.Markdown(
                "Only labels, scores, match counts, and execution mode are retained in this "
                "browser session (maximum 10 rows). Raw submitted text is not retained here."
            )
            history_output = gr.Dataframe(
                headers=[
                    "entry",
                    "top emotion",
                    "emotion score",
                    "sentiment",
                    "sentiment score",
                    "match categories",
                    "mode",
                ],
                datatype=["number", "str", "number", "str", "number", "number", "str"],
                interactive=False,
                label="Per-session aggregate history",
            )
        with gr.Tab("Machine-readable report"):
            report_output = gr.JSON(label="Analysis metadata (submitted text excluded)")

        gr.Markdown(
            "#### Known limitations\n" + "\n".join(f"- {item}" for item in LIMITATIONS),
            elem_classes=["sentinex-limitations"],
        )

        outputs = [
            status_output,
            emotion_output,
            sentiment_output,
            lexical_output,
            history_output,
            report_output,
            session_history,
        ]
        analyze_button.click(
            fn=analyze_text,
            inputs=[text_input, session_history],
            outputs=outputs,
        )
        clear_button.click(
            fn=clear_session,
            inputs=None,
            outputs=[text_input, *outputs],
        )

    return demo


def main() -> None:
    host = os.getenv("SENTINEX_HOST", "127.0.0.1")
    theme = gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="sky",
        neutral_hue="slate",
        radius_size="lg",
    )
    build_app().launch(
        server_name=host,
        server_port=7860,
        theme=theme,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
