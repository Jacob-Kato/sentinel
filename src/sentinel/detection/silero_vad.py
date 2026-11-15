from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SileroVADFrame:
    timestamp_seconds: float
    speech_probability: float
    is_speech: bool
    duration_seconds: float


class SileroVAD:
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 512
    CONTEXT_SIZE = 64

    def __init__(
        self,
        threshold: float = 0.5,
        model_path: str | Path | None = None,
        session: Any | None = None,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")

        self.threshold = threshold

        if session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise ImportError(
                    "Silero VAD requires ONNX Runtime. "
                    "Install it with: pip install -e '.[vad-silero]'"
                ) from exc

            if model_path is None:
                model_path = (
                    Path(__file__).resolve().parents[3]
                    / "models"
                    / "silero_vad.onnx"
                )

            model_path = Path(model_path)

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Silero VAD model not found: {model_path}"
                )

            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )

        self.session = session
        self.reset_stream()

    def reset_stream(self) -> None:
        """Start a new independent stream, clearing model and audio state."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.CONTEXT_SIZE), dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._samples_processed = 0

    def _validate_audio(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
        *,
        allow_empty: bool = False,
    ) -> npt.NDArray[np.float32]:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("Sample rate must be an integer.")

        if sample_rate != self.SAMPLE_RATE:
            raise ValueError("Silero VAD requires a 16000 Hz sample rate.")

        if not isinstance(audio, np.ndarray):
            raise TypeError("Audio must be a NumPy array.")

        if audio.ndim != 1:
            raise ValueError("Audio must be mono (one-dimensional).")

        if audio.size == 0 and not allow_empty:
            raise ValueError("Audio cannot be empty.")

        if not np.issubdtype(audio.dtype, np.floating):
            raise TypeError("Audio must contain floating-point samples.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains non-finite values.")

        if np.any(np.abs(audio) > 1.0):
            raise ValueError("Audio samples must be between -1.0 and 1.0.")

        return audio.astype(np.float32, copy=False)

    def _infer_frame(
        self,
        chunk: npt.NDArray[np.float32],
        timestamp_seconds: float,
        valid_samples: int,
    ) -> SileroVADFrame:
        model_input = np.concatenate((self._context, chunk.reshape(1, -1)), axis=1)
        output, self._state = self.session.run(
            None,
            {
                "input": model_input.astype(np.float32, copy=False),
                "state": self._state,
                "sr": np.array(self.SAMPLE_RATE, dtype=np.int64),
            },
        )

        probability = float(output[0][0])
        if not 0.0 <= probability <= 1.0:
            raise RuntimeError(f"Model returned invalid probability: {probability}")

        self._context = chunk.reshape(1, -1)[:, -self.CONTEXT_SIZE :].copy()
        return SileroVADFrame(
            timestamp_seconds=timestamp_seconds,
            speech_probability=probability,
            is_speech=probability >= self.threshold,
            duration_seconds=valid_samples / self.SAMPLE_RATE,
        )

    def detect_stream(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> list[SileroVADFrame]:
        """Process an audio block while retaining state across consecutive calls.

        Only complete 512-sample frames are inferred. Any remainder is buffered
        until the next call; call ``reset_stream`` before a new independent
        stream to discard that remainder and clear recurrent model state.
        """
        audio = self._validate_audio(audio, sample_rate, allow_empty=True)
        if audio.size == 0:
            return []

        self._pending = np.concatenate((self._pending, audio))
        results = []
        while self._pending.size >= self.CHUNK_SIZE:
            chunk = self._pending[: self.CHUNK_SIZE]
            self._pending = self._pending[self.CHUNK_SIZE :]
            results.append(
                self._infer_frame(
                    chunk,
                    self._samples_processed / self.SAMPLE_RATE,
                    self.CHUNK_SIZE,
                )
            )
            self._samples_processed += self.CHUNK_SIZE
        return results

    def flush_stream(self) -> list[SileroVADFrame]:
        """Process any final partial frame when a stream is ending."""
        if self._pending.size == 0:
            return []

        valid_samples = self._pending.size
        chunk = np.pad(self._pending, (0, self.CHUNK_SIZE - valid_samples))
        self._pending = np.empty(0, dtype=np.float32)
        frame = self._infer_frame(
            chunk,
            self._samples_processed / self.SAMPLE_RATE,
            valid_samples,
        )
        self._samples_processed += valid_samples
        return [frame]

    def detect(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> list[SileroVADFrame]:
        audio = self._validate_audio(audio, sample_rate)
        self.reset_stream()
        results = []

        for start in range(0, len(audio), self.CHUNK_SIZE):
            chunk = audio[start : start + self.CHUNK_SIZE]
            valid_samples = len(chunk)

            if valid_samples < self.CHUNK_SIZE:
                chunk = np.pad(
                    chunk,
                    (0, self.CHUNK_SIZE - valid_samples),
                )

            chunk = chunk.reshape(1, self.CHUNK_SIZE)

            results.append(
                self._infer_frame(
                    chunk,
                    start / self.SAMPLE_RATE,
                    valid_samples,
                )
            )
            self._samples_processed += valid_samples

        return results
