import wave
from pathlib import Path

import numpy as np

from sentinel.detection.webrtc_vad import WebRTCVAD


SAMPLE_RATE = 16000
FRAME_DURATION = 20

vad = WebRTCVAD(
    aggressiveness=2,
    frame_duration_ms=FRAME_DURATION,
)


def load_audio(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise ValueError(f"{path} must be mono.")

        if wav.getsampwidth() != 2:
            raise ValueError(f"{path} must be 16-bit PCM.")

        if wav.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{path} must be 16 kHz.")

        audio_bytes = wav.readframes(wav.getnframes())

    return (
        np.frombuffer(audio_bytes, dtype=np.int16)
        .astype(np.float32)
        / 32768.0
    )


def evaluate(path: Path):
    audio = load_audio(path)

    samples_per_frame = SAMPLE_RATE * FRAME_DURATION // 1000
    usable_samples = (
        len(audio) // samples_per_frame
    ) * samples_per_frame

    audio = audio[:usable_samples]

    results = vad.detect(audio, SAMPLE_RATE)

    speech_frames = sum(result.is_speech for result in results)
    total_frames = len(results)

    speech_percentage = speech_frames / total_frames * 100

    print(f"\nRecording: {path.name}")
    print(f"Duration: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"Total frames: {total_frames}")
    print(f"Speech frames: {speech_frames}")
    print(f"Non-speech frames: {total_frames - speech_frames}")
    print(f"Speech-positive percentage: {speech_percentage:.2f}%")


for filename in ("silence.wav", "speech.wav", "environment.wav"):
    evaluate(Path("test_audio") / filename)
