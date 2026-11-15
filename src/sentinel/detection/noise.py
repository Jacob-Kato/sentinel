"""Low-cost sound activity detection for the first alert tier."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sentinel.detection.alerts import AlertEvent, AlertTier


class NoiseActivityTracker:
    """Emit one level-one alert when sound crosses an RMS threshold."""

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold_rms: float = 0.015,
        frame_samples: int = 512,
        start_frames: int = 3,
        end_frames: int = 24,
    ) -> None:
        if sample_rate <= 0 or frame_samples <= 0:
            raise ValueError("Sample rate and frame size must be positive.")
        if not 0 < threshold_rms <= 1:
            raise ValueError("RMS threshold must be greater than zero and at most 1.")
        if start_frames < 1 or end_frames < 1:
            raise ValueError("Transition frame counts must be positive.")
        self.sample_rate = sample_rate
        self.threshold_rms = threshold_rms
        self.frame_samples = frame_samples
        self.start_frames = start_frames
        self.end_frames = end_frames
        self._buffer = np.empty(0, dtype=np.float32)
        self._buffer_start_seconds = 0.0
        self._active = False
        self._above_run = 0
        self._below_run = 0
        self._candidate_timestamp = 0.0

    def update(
        self,
        audio: npt.NDArray[np.float32],
        timestamp_seconds: float,
    ) -> list[AlertEvent]:
        if audio.ndim != 1:
            raise ValueError("Audio must be one-dimensional.")
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid values.")
        if not audio.size:
            return []
        if not self._buffer.size:
            self._buffer_start_seconds = timestamp_seconds
        self._buffer = np.concatenate(
            (self._buffer, audio.astype(np.float32, copy=False))
        )
        events: list[AlertEvent] = []
        while self._buffer.size >= self.frame_samples:
            frame = self._buffer[: self.frame_samples]
            self._buffer = self._buffer[self.frame_samples :]
            frame_start = self._buffer_start_seconds
            self._buffer_start_seconds += self.frame_samples / self.sample_rate
            rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
            if rms >= self.threshold_rms:
                self._below_run = 0
                if not self._active:
                    if self._above_run == 0:
                        self._candidate_timestamp = frame_start
                    self._above_run += 1
                    if self._above_run >= self.start_frames:
                        self._active = True
                        self._above_run = 0
                        events.append(
                            AlertEvent(
                                AlertTier.NOISE,
                                "Noise detected.",
                                self._candidate_timestamp,
                            )
                        )
                else:
                    self._below_run = 0
            else:
                self._above_run = 0
                if self._active:
                    self._below_run += 1
                    if self._below_run >= self.end_frames:
                        self._active = False
                        self._below_run = 0
        return events
