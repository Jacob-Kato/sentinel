"""Streaming YAMNet alarm and siren classifier backed by LiteRT."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from sentinel.detection.alerts import AlertEvent, AlertTier


SAMPLE_RATE = 16000
WINDOW_SAMPLES = 15600  # 975 ms, the minimum YAMNet input window.
HOP_SAMPLES = 7680  # 480 ms, matching YAMNet's 50% overlap.
MODEL_PATH = Path("models/yamnet.tflite")
EMERGENCY_CLASSES = {
    390: "siren",
    391: "civil defense siren",
    393: "smoke detector alarm",
    394: "fire alarm",
}
ATTENTION_CLASSES = {353: "knock"}
SOUND_CLASSES = {**ATTENTION_CLASSES, **EMERGENCY_CLASSES}


class YamnetAlarmDetector:
    """Classify streamed audio windows and latch repeated emergency sounds."""

    def __init__(
        self,
        interpreter: Any,
        threshold: float = 0.25,
        confirmation_windows: int = 2,
        clear_windows: int = 4,
        knock_threshold: float = 0.4,
        detect_knocks: bool = True,
    ) -> None:
        if not 0 < threshold <= 1:
            raise ValueError("Alarm confidence threshold must be in (0, 1].")
        if not 0 < knock_threshold <= 1:
            raise ValueError("Knock confidence threshold must be in (0, 1].")
        if not 1 <= confirmation_windows <= 3 or clear_windows < 1:
            raise ValueError(
                "Confirmation windows must be 1-3 and clear windows must be positive."
            )
        self.interpreter = interpreter
        self.threshold = threshold
        self.knock_threshold = knock_threshold
        self.detect_knocks = detect_knocks
        self.confirmation_windows = confirmation_windows
        self.clear_windows = clear_windows
        self.interpreter.allocate_tensors()
        self.input_detail: dict[str, Any]
        self.output_details: list[dict[str, Any]]
        self.input_detail = interpreter.get_input_details()[0]
        self.output_details = interpreter.get_output_details()
        self._recent: deque[tuple[AlertTier, int, float] | None] = deque(maxlen=3)
        self._active_detection: tuple[AlertTier, int] | None = None
        self._clear_run = 0

    @classmethod
    def from_model(
        cls,
        path: str | Path = MODEL_PATH,
        threshold: float = 0.25,
        knock_threshold: float = 0.4,
        detect_knocks: bool = True,
    ) -> YamnetAlarmDetector:
        model_path = Path(path)
        if not model_path.is_file():
            raise RuntimeError(
                f"YAMNet model not found at {model_path}. "
                "Run ./run_sentinel.sh to download it."
            )
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise RuntimeError(
                "Alarm sound classification needs LiteRT. Run ./run_sentinel.sh to install it."
            ) from exc
        return cls(
            Interpreter(model_path=str(model_path), num_threads=2),
            threshold=threshold,
            knock_threshold=knock_threshold,
            detect_knocks=detect_knocks,
        )

    @classmethod
    def from_environment(cls) -> YamnetAlarmDetector:
        path = os.environ.get("SENTINEL_YAMNET_MODEL_PATH", MODEL_PATH)
        threshold = float(os.environ.get("SENTINEL_ALARM_THRESHOLD", "0.25"))
        knock_threshold = float(os.environ.get("SENTINEL_KNOCK_THRESHOLD", "0.4"))
        detect_knocks = os.environ.get("SENTINEL_DETECT_KNOCKS", "1").casefold()
        return cls.from_model(
            path,
            threshold,
            knock_threshold,
            detect_knocks not in {"0", "false", "no", "off"},
        )

    def classify_window(
        self,
        audio: npt.NDArray[np.float32],
        timestamp_seconds: float,
    ) -> AlertEvent | None:
        if audio.ndim != 1 or audio.size != WINDOW_SAMPLES:
            raise ValueError(
                f"YAMNet input must contain {WINDOW_SAMPLES} mono samples."
            )
        tensor = np.asarray(audio, dtype=np.float32).reshape(
            self.input_detail["shape"]
        )
        dtype = np.dtype(self.input_detail["dtype"])
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self.input_detail.get("quantization", (0.0, 0))
            if scale:
                tensor = np.rint(tensor / scale + zero_point)
            tensor = np.clip(
                tensor,
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
            ).astype(dtype)
        else:
            tensor = tensor.astype(dtype, copy=False)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        scores = self._read_scores()
        candidates: list[tuple[AlertTier, int, float]] = []
        for index in SOUND_CLASSES:
            if index in ATTENTION_CLASSES and not self.detect_knocks:
                continue
            if index >= scores.size:
                continue
            confidence = float(scores[index])
            tier = (
                AlertTier.ATTENTION
                if index in ATTENTION_CLASSES
                else AlertTier.EMERGENCY
            )
            threshold = (
                self.knock_threshold
                if tier is AlertTier.ATTENTION
                else self.threshold
            )
            if confidence >= threshold:
                candidates.append((tier, index, confidence))
        candidate = (
            max(candidates, key=lambda item: (item[0], item[2]))
            if candidates
            else None
        )
        self._recent.append(candidate)

        if self._active_detection is not None:
            if (
                candidate is not None
                and candidate[:2] == self._active_detection
            ):
                self._clear_run = 0
            else:
                self._clear_run += 1
                if self._clear_run >= self.clear_windows:
                    self._active_detection = None
                    self._clear_run = 0
        if self._active_detection is not None:
            return None

        counts: dict[tuple[AlertTier, int], list[float]] = {}
        for recent in self._recent:
            if recent is not None:
                key = (recent[0], recent[1])
                counts.setdefault(key, []).append(recent[2])
        confirmed = [
            (key, max(confidences))
            for key, confidences in counts.items()
            if len(confidences) >= self.confirmation_windows
        ]
        if not confirmed:
            return None
        (tier, index), confidence = max(
            confirmed,
            key=lambda item: (item[0][0], item[1]),
        )
        self._active_detection = (tier, index)
        self._clear_run = 0
        sound_name = SOUND_CLASSES[index]
        message = (
            "Knock detected."
            if tier is AlertTier.ATTENTION
            else f"Emergency sound detected: {sound_name}."
        )
        return AlertEvent(
            tier,
            message,
            timestamp_seconds,
            confidence,
        )

    def reset(self) -> None:
        """Clear temporal alarm state after audio frames have been dropped."""
        self._recent.clear()
        self._active_detection = None
        self._clear_run = 0

    def _read_scores(self) -> npt.NDArray[np.float32]:
        for detail in self.output_details:
            values = np.asarray(self.interpreter.get_tensor(detail["index"]))
            if values.size >= max(EMERGENCY_CLASSES) + 1:
                dtype = np.dtype(detail["dtype"])
                if np.issubdtype(dtype, np.integer):
                    scale, zero_point = detail.get("quantization", (0.0, 0))
                    if scale:
                        values = (values.astype(np.float32) - zero_point) * scale
                return values.reshape(-1).astype(np.float32, copy=False)
        raise RuntimeError("YAMNet output does not contain the expected 521 classes.")


class YamnetStream:
    """Assemble overlapping YAMNet windows without saving audio to disk."""

    def __init__(self, detector: YamnetAlarmDetector) -> None:
        self.detector = detector
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_seconds = 0.0

    def push(
        self,
        audio: npt.NDArray[np.float32],
        timestamp_seconds: float,
    ) -> list[AlertEvent]:
        if audio.ndim != 1:
            raise ValueError("Audio must be one-dimensional.")
        if not audio.size:
            return []
        if self._buffer.size:
            expected_timestamp = (
                self._buffer_start_seconds + self._buffer.size / SAMPLE_RATE
            )
            if abs(timestamp_seconds - expected_timestamp) > 0.01:
                self.reset()
        if not self._buffer.size:
            self._buffer_start_seconds = timestamp_seconds
        self._buffer = np.concatenate(
            (self._buffer, audio.astype(np.float32, copy=False))
        )
        events: list[AlertEvent] = []
        while self._buffer.size >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            event = self.detector.classify_window(window, self._buffer_start_seconds)
            if event is not None:
                events.append(event)
            self._buffer = self._buffer[HOP_SAMPLES:]
            self._buffer_start_seconds += HOP_SAMPLES / SAMPLE_RATE
        return events

    def reset(self) -> None:
        """Discard buffered audio and alarm state across a dropped-frame gap."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_seconds = 0.0
        self.detector.reset()
