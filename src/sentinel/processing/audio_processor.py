from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AudioMeasurements:
    peak_amplitude: float
    rms_amplitude: float
    clipping_ratio: float
    sample_count: int
    duration_seconds: float
    captured_at: datetime


@dataclass(frozen=True)
class AudioProcessingResult:
    original_audio: npt.NDArray[np.float32]
    measurements: AudioMeasurements


class AudioProcessor:
    def __init__(
        self,
        clipping_threshold: float = 0.999,
    ):
        if not 0.0 < clipping_threshold <= 1.0:
            raise ValueError("Clipping threshold must be between 0 and 1.")

        self.clipping_threshold = clipping_threshold

    def process(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> AudioProcessingResult:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if audio.ndim != 1:
            raise ValueError("Audio must be a one-dimensional array.")

        if audio.size == 0:
            raise ValueError("Audio cannot be empty.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid values.")

        original_audio = np.array(audio, dtype=np.float32, copy=True)

        peak_amplitude = float(np.max(np.abs(original_audio)))
        rms_amplitude = float(np.sqrt(np.mean(original_audio**2)))

        clipping_ratio = float(
            np.mean(np.abs(original_audio) >= self.clipping_threshold)
        )

        measurements = AudioMeasurements(
            peak_amplitude=peak_amplitude,
            rms_amplitude=rms_amplitude,
            clipping_ratio=clipping_ratio,
            sample_count=int(original_audio.size),
            duration_seconds=original_audio.size / sample_rate,
            captured_at=datetime.now(UTC),
        )

        return AudioProcessingResult(
            original_audio=original_audio,
            measurements=measurements,
        )
