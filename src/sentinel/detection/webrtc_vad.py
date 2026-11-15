from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import webrtcvad


@dataclass(frozen=True)
class VoiceActivityFrame:
    timestamp_seconds: float
    is_speech: bool


class WebRTCVAD:
    SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
    SUPPORTED_FRAME_DURATIONS = (10, 20, 30)

    def __init__(
        self,
        aggressiveness: int = 2,
        frame_duration_ms: int = 30,
    ):
        if aggressiveness not in range(4):
            raise ValueError("Aggressiveness must be between 0 and 3.")

        if frame_duration_ms not in self.SUPPORTED_FRAME_DURATIONS:
            raise ValueError("Frame duration must be 10, 20, or 30 ms.")

        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_duration_ms = frame_duration_ms

    def detect(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> list[VoiceActivityFrame]:

        if (
            not isinstance(sample_rate, int)
            or isinstance(sample_rate, bool)
            or sample_rate not in self.SUPPORTED_SAMPLE_RATES):
            raise ValueError("Unsupported sample rate.")

        if audio.ndim != 1:
            raise ValueError("Audio must be mono.")
        if not np.issubdtype(audio.dtype, np.floating):
            raise ValueError("Audio must contain floating-point samples.")

        if audio.size == 0:
            raise ValueError("Audio cannot be empty.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid values.")

        if np.any(np.abs(audio) > 1.0):
            raise ValueError("Audio samples must be between -1 and 1.")

        samples_per_frame = (
            sample_rate * self.frame_duration_ms // 1000
        )

        if audio.size % samples_per_frame != 0:
            raise ValueError(
                "Audio length must contain complete VAD frames."
            )

        pcm_audio = (
            np.clip(audio, -1.0, 1.0) * 32767
        ).astype(np.int16)

        results = []

        for index, start in enumerate(
            range(0, pcm_audio.size, samples_per_frame)
        ):
            frame = pcm_audio[start : start + samples_per_frame]

            speech_detected = self.vad.is_speech(
                frame.tobytes(),
                sample_rate,
            )

            results.append(
                VoiceActivityFrame(
                    timestamp_seconds=(
                        index * self.frame_duration_ms / 1000
                    ),
                    is_speech=speech_detected,
                )
            )

        return results
