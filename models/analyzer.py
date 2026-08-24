"""Session-safe language-signal analysis for the SENTINEX research demo.

The module deliberately does not diagnose, predict crisis, or infer whether a
person is safe. It exposes model labels and literal phrase matches so their
limitations can be inspected.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Protocol

MAX_INPUT_CHARACTERS = 2_000
MAX_HISTORY_ENTRIES = 10

EMOTION_LABELS = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)
SENTIMENT_LABELS = ("negative", "neutral", "positive")

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
EMOTION_REVISION = "0e1cd914e3d46199ed785853e12b57304e04178b"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
SENTIMENT_REVISION = "3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7"

SAFETY_NOTICE = (
    "SENTINEX cannot determine whether anyone is safe. If you may act on thoughts "
    "of self-harm or someone is in immediate danger, contact local emergency "
    "services now. In India, the Ministry of Health and Family Welfare operates "
    "Tele-MANAS: https://telemanas.mohfw.gov.in/."
)

LIMITATIONS = (
    "Classifier labels are not diagnoses, clinical measurements, or calibrated safety scores.",
    "Literal phrase matching does not understand negation, quotation, context, or intent.",
    "Demo distributions are synthetic and test application plumbing only.",
    "No accuracy, fairness, crisis-detection, or clinical-validation claim is established.",
)

LEXICAL_CATEGORIES: dict[str, tuple[str, ...]] = {
    "support_related_wording": (
        "want to die",
        "end it all",
        "no reason to live",
        "better off dead",
        "give up on life",
        "disappear forever",
    ),
    "social_disconnection_wording": (
        "feel alone",
        "nobody cares",
        "no one cares",
        "feel isolated",
        "nobody understands",
    ),
    "sleep_or_energy_wording": (
        "can't sleep",
        "cant sleep",
        "cry myself to sleep",
        "tired all the time",
        "no energy",
    ),
    "negative_self_reference": (
        "my fault",
        "blame myself",
        "i am useless",
        "hate myself",
        "all my fault",
    ),
    "contrast_cue": (
        "yeah right",
        "just perfect",
        "oh great",
        "couldn't be better",
        "couldnt be better",
    ),
}


class ExecutionMode(StrEnum):
    DEMO = "demo"
    TRANSFORMERS = "transformers"


@dataclass(frozen=True, slots=True)
class Classification:
    label: str
    score: float

    def to_mapping(self) -> dict[str, str | float]:
        return {"label": self.label, "score": self.score}


@dataclass(frozen=True, slots=True)
class BackendOutput:
    emotions: tuple[Classification, ...]
    sentiments: tuple[Classification, ...]


@dataclass(frozen=True, slots=True)
class LexicalSignal:
    category: str
    phrases: tuple[str, ...]

    def to_mapping(self) -> dict[str, str | list[str]]:
        return {"category": self.category, "literal_matches": list(self.phrases)}


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    top_emotion_label: str
    top_emotion_score: float
    sentiment_label: str
    sentiment_score: float
    lexical_category_count: int
    execution_mode: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> HistoryEntry:
        return cls(
            top_emotion_label=str(value["top_emotion_label"]),
            top_emotion_score=float(value["top_emotion_score"]),
            sentiment_label=str(value["sentiment_label"]),
            sentiment_score=float(value["sentiment_score"]),
            lexical_category_count=int(value["lexical_category_count"]),
            execution_mode=str(value["execution_mode"]),
        )

    def to_mapping(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    execution_mode: ExecutionMode
    evidence_status: str
    emotions: tuple[Classification, ...]
    sentiments: tuple[Classification, ...]
    lexical_signals: tuple[LexicalSignal, ...]
    history: tuple[HistoryEntry, ...]
    backend_details: Mapping[str, str]
    limitations: tuple[str, ...] = LIMITATIONS
    safety_notice: str = SAFETY_NOTICE

    @property
    def top_emotion(self) -> Classification:
        return self.emotions[0]

    @property
    def top_sentiment(self) -> Classification:
        return self.sentiments[0]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode.value,
            "evidence_status": self.evidence_status,
            "emotions": [item.to_mapping() for item in self.emotions],
            "sentiments": [item.to_mapping() for item in self.sentiments],
            "lexical_signals": [item.to_mapping() for item in self.lexical_signals],
            "history": [item.to_mapping() for item in self.history],
            "backend_details": dict(self.backend_details),
            "limitations": list(self.limitations),
            "safety_notice": self.safety_notice,
        }


class ClassifierBackend(Protocol):
    mode: ExecutionMode
    evidence_status: str
    details: Mapping[str, str]

    def classify(self, text: str) -> BackendOutput: ...


class DemoBackend:
    """Deterministic, input-dependent synthetic classifier for offline tests."""

    mode = ExecutionMode.DEMO
    evidence_status = "synthetic"
    details: Mapping[str, str] = {
        "backend": "deterministic SHA-256 fixture",
        "network": "disabled",
    }

    @staticmethod
    def _distribution(
        text: str,
        namespace: str,
        labels: Sequence[str],
    ) -> tuple[Classification, ...]:
        values = []
        for label in labels:
            digest = hashlib.sha256(f"{namespace}\0{label}\0{text}".encode()).digest()
            values.append(int.from_bytes(digest[:4], "big") + 1)
        total = sum(values)
        result = [
            Classification(label=label, score=round(value / total, 6))
            for label, value in zip(labels, values, strict=True)
        ]
        return tuple(sorted(result, key=lambda item: item.score, reverse=True))

    def classify(self, text: str) -> BackendOutput:
        return BackendOutput(
            emotions=self._distribution(text, "emotion", EMOTION_LABELS),
            sentiments=self._distribution(text, "sentiment", SENTIMENT_LABELS),
        )


PipelineFactory = Callable[..., Any]


class TransformersBackend:
    """Pinned Hugging Face text classifiers with validated label semantics."""

    mode = ExecutionMode.TRANSFORMERS
    evidence_status = "model_output"
    details: Mapping[str, str] = {
        "emotion_model": f"{EMOTION_MODEL}@{EMOTION_REVISION}",
        "sentiment_model": f"{SENTIMENT_MODEL}@{SENTIMENT_REVISION}",
        "clinical_validation": "not established",
    }

    def __init__(self, pipeline_factory: PipelineFactory | None = None) -> None:
        if pipeline_factory is None:
            try:
                from transformers import pipeline as pipeline_factory
            except ImportError as error:
                raise RuntimeError(
                    "Transformer mode requires the optional 'models' dependency group."
                ) from error

        self._emotion = pipeline_factory(
            "text-classification",
            model=EMOTION_MODEL,
            revision=EMOTION_REVISION,
            trust_remote_code=False,
        )
        self._sentiment = pipeline_factory(
            "text-classification",
            model=SENTIMENT_MODEL,
            revision=SENTIMENT_REVISION,
            trust_remote_code=False,
        )
        self._inference_lock = threading.Lock()

    @staticmethod
    def _validated_distribution(
        raw: Any,
        expected_labels: Sequence[str],
    ) -> tuple[Classification, ...]:
        while isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list):
            raw = raw[0]
        if not isinstance(raw, list) or not raw:
            raise ValueError("Classifier returned an unexpected empty or non-list result.")

        values = []
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("Classifier returned a non-mapping label record.")
            label = str(item.get("label", "")).casefold()
            score = float(item.get("score", -1.0))
            if label not in expected_labels or not 0.0 <= score <= 1.0:
                raise ValueError(f"Classifier returned unsupported label semantics: {label!r}")
            values.append(Classification(label=label, score=round(score, 6)))

        observed = {item.label for item in values}
        if observed != set(expected_labels):
            raise ValueError(
                f"Classifier labels do not match the pinned model contract: {sorted(observed)}"
            )
        return tuple(sorted(values, key=lambda item: item.score, reverse=True))

    def classify(self, text: str) -> BackendOutput:
        with self._inference_lock:
            emotions = self._emotion(text, top_k=None, truncation=True, max_length=512)
            sentiments = self._sentiment(text, top_k=None, truncation=True, max_length=512)
        return BackendOutput(
            emotions=self._validated_distribution(emotions, EMOTION_LABELS),
            sentiments=self._validated_distribution(sentiments, SENTIMENT_LABELS),
        )


class SentinexAnalyzer:
    """Stateless analyzer; session history is supplied and returned by the caller."""

    def __init__(self, backend: ClassifierBackend | None = None) -> None:
        self._backend = backend or DemoBackend()

    @staticmethod
    def _normalise_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("’", "'").strip()).casefold()

    @classmethod
    def _literal_signals(cls, text: str) -> tuple[LexicalSignal, ...]:
        normalised = cls._normalise_text(text)
        signals = []
        for category, phrases in LEXICAL_CATEGORIES.items():
            matches = []
            for phrase in phrases:
                escaped = re.escape(phrase).replace(r"\ ", r"\s+")
                if re.search(rf"(?<!\w){escaped}(?!\w)", normalised):
                    matches.append(phrase)
            if matches:
                signals.append(LexicalSignal(category=category, phrases=tuple(matches)))
        return tuple(signals)

    def analyze(
        self,
        text: str,
        history: Sequence[HistoryEntry | Mapping[str, Any]] | None = None,
    ) -> AnalysisResult:
        if not isinstance(text, str):
            raise TypeError("Input must be text.")
        cleaned = text.strip()
        if len(cleaned) < 5:
            raise ValueError("Enter at least 5 characters.")
        if len(cleaned) > MAX_INPUT_CHARACTERS:
            raise ValueError(f"Input must not exceed {MAX_INPUT_CHARACTERS} characters.")

        existing = []
        for item in history or ():
            existing.append(
                item if isinstance(item, HistoryEntry) else HistoryEntry.from_mapping(item)
            )

        backend_output = self._backend.classify(cleaned)
        signals = self._literal_signals(cleaned)
        top_emotion = backend_output.emotions[0]
        top_sentiment = backend_output.sentiments[0]
        next_entry = HistoryEntry(
            top_emotion_label=top_emotion.label,
            top_emotion_score=top_emotion.score,
            sentiment_label=top_sentiment.label,
            sentiment_score=top_sentiment.score,
            lexical_category_count=len(signals),
            execution_mode=self._backend.mode.value,
        )
        next_history = tuple((existing + [next_entry])[-MAX_HISTORY_ENTRIES:])

        return AnalysisResult(
            execution_mode=self._backend.mode,
            evidence_status=self._backend.evidence_status,
            emotions=backend_output.emotions,
            sentiments=backend_output.sentiments,
            lexical_signals=signals,
            history=next_history,
            backend_details=self._backend.details,
        )


def build_analyzer(mode: ExecutionMode | str | None = None) -> SentinexAnalyzer:
    configured = ExecutionMode(mode or os.getenv("SENTINEX_MODE", ExecutionMode.DEMO.value))
    if configured == ExecutionMode.DEMO:
        return SentinexAnalyzer(DemoBackend())
    return SentinexAnalyzer(TransformersBackend())
