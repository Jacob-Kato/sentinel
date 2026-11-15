"""Offline phrase spotting using Vosk's constrained English vocabulary."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sentinel.detection.alerts import AlertEvent, AlertTier


SAMPLE_RATE = 16000
MODEL_DIRECTORY = Path("models/vosk-model-small-en-us-0.15")
GRAMMAR = [
    "hey",
    "excuse me",
    "help",
    "help me",
    "emergency",
    "fire alarm",
    "smoke alarm",
]


class VoskPhraseDetector:
    """Recognize only configured attention and emergency phrases, locally."""

    def __init__(self, model_path: str | Path, model: Any | None = None) -> None:
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            raise RuntimeError(
                "Phrase recognition needs Vosk. Run ./run_sentinel.sh to install it."
            ) from exc

        path = Path(model_path)
        if not path.is_dir():
            raise RuntimeError(
                f"Vosk model not found at {path}. Run ./run_sentinel.sh to download it."
            )
        SetLogLevel(-1)
        self._model = model if model is not None else Model(str(path))
        self._recognizer_class = KaldiRecognizer
        self._recognizer = self._new_recognizer()
        self._seen_tiers_this_utterance: set[AlertTier] = set()

    @classmethod
    def from_environment(cls) -> VoskPhraseDetector:
        configured = os.environ.get("SENTINEL_VOSK_MODEL_PATH")
        return cls(configured or MODEL_DIRECTORY)

    def _new_recognizer(self) -> Any:
        recognizer = self._recognizer_class(
            self._model,
            SAMPLE_RATE,
            json.dumps(GRAMMAR),
        )
        recognizer.SetWords(False)
        return recognizer

    def process(
        self,
        audio: npt.NDArray[np.float32],
        timestamp_seconds: float,
    ) -> list[AlertEvent]:
        if audio.ndim != 1:
            raise ValueError("Audio must be one-dimensional.")
        if not audio.size:
            return []
        pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        is_final = self._recognizer.AcceptWaveform(pcm)
        result_text = (
            self._recognizer.Result()
            if is_final
            else self._recognizer.PartialResult()
        )
        payload = json.loads(result_text)
        text = payload.get("text", "") if is_final else payload.get("partial", "")
        events = self._events_for_text(text, timestamp_seconds)
        if is_final:
            self.reset()
        return events

    def reset(self) -> None:
        """Discard partial phrase state after a stream discontinuity."""
        self._recognizer = self._new_recognizer()
        self._seen_tiers_this_utterance.clear()

    def _events_for_text(self, text: str, timestamp_seconds: float) -> list[AlertEvent]:
        normalized = re.sub(r"[^a-z ]", " ", text.casefold())
        normalized = " ".join(normalized.split())
        emergency_phrases = {
            "help",
            "help me",
            "emergency",
            "fire alarm",
            "smoke alarm",
        }
        matches = [
            phrase
            for phrase in sorted(GRAMMAR, key=len, reverse=True)
            if re.search(rf"\b{re.escape(phrase)}\b", normalized)
        ]
        if not matches:
            return []

        # If a phrase such as "help me" is recognized, avoid treating its
        # shorter component "help" as a second alert in the same utterance.
        matches = [
            phrase
            for phrase in matches
            if not any(
                phrase != other
                and re.search(rf"\b{re.escape(phrase)}\b", other)
                for other in matches
            )
        ]
        matches.sort(key=lambda phrase: phrase not in emergency_phrases)
        phrase = next(
            (
                item
                for item in matches
                if (
                    AlertTier.EMERGENCY
                    if item in emergency_phrases
                    else AlertTier.ATTENTION
                )
                not in self._seen_tiers_this_utterance
            ),
            None,
        )
        if phrase is None:
            return []
        tier = (
            AlertTier.EMERGENCY
            if phrase in emergency_phrases
            else AlertTier.ATTENTION
        )
        label = (
            "Emergency phrase"
            if tier is AlertTier.EMERGENCY
            else "Attention phrase"
        )
        self._seen_tiers_this_utterance.add(tier)
        return [
            AlertEvent(tier, f"{label} detected: ‘{phrase}’.", timestamp_seconds)
        ]
