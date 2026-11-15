import wave
from pathlib import Path

import numpy as np

from sentinel.detection.silero_vad import SileroVAD
from sentinel.detection.webrtc_vad import WebRTCVAD

SAMPLE_RATE = 16000
WEBRTC_FRAME_DURATION = 0.020

RECORDINGS = Path("test_audio/environmental")
CONTROLLED_FILE = RECORDINGS / "controlled_speech.wav"

SPEECH_INTERVALS = [
    (5.0, 10.0),
    (15.0, 20.0),
    (25.0, 30.0),
]

EVALUATION_END = 30.0

silero = SileroVAD(threshold=0.5)
webrtc = WebRTCVAD(aggressiveness=2, frame_duration_ms=20)


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


def is_ground_truth_speech(timestamp: float) -> bool:
    return any(
        start <= timestamp < end
        for start, end in SPEECH_INTERVALS
    )


def calculate_metrics(frames, frame_duration: float) -> dict:
    tp = fp = tn = fn = 0.0

    for frame in frames:
        duration = getattr(frame, "duration_seconds", frame_duration)
        midpoint = frame.timestamp_seconds + duration / 2

        if midpoint >= EVALUATION_END:
            continue

        actual_speech = is_ground_truth_speech(midpoint)
        predicted_speech = frame.is_speech

        if actual_speech and predicted_speech:
            tp += duration
        elif not actual_speech and predicted_speech:
            fp += duration
        elif not actual_speech and not predicted_speech:
            tn += duration
        else:
            fn += duration

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )
    false_positive_rate = (
        fp / (fp + tn) if fp + tn else 0
    )

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{name}")
    print("-" * 35)
    print(f"True positives: {metrics['TP']:.2f}s")
    print(f"False positives: {metrics['FP']:.2f}s")
    print(f"True negatives: {metrics['TN']:.2f}s")
    print(f"False negatives: {metrics['FN']:.2f}s")
    print(f"Precision: {metrics['precision'] * 100:.2f}%")
    print(f"Recall: {metrics['recall'] * 100:.2f}%")
    print(f"F1 score: {metrics['f1'] * 100:.2f}%")
    print(
        f"False-positive rate: "
        f"{metrics['false_positive_rate'] * 100:.2f}%"
    )


def evaluate_environment(path: Path) -> None:
    audio = load_audio(path)

    silero_results = silero.detect(audio, SAMPLE_RATE)
    webrtc_results = webrtc.detect(audio, SAMPLE_RATE)

    silero_duration = sum(
        frame.duration_seconds for frame in silero_results
    )

    webrtc_duration = len(webrtc_results) * WEBRTC_FRAME_DURATION

    silero_speech = sum(
        frame.duration_seconds
        for frame in silero_results
        if frame.is_speech
    )

    webrtc_speech = sum(
        frame.is_speech for frame in webrtc_results
    ) * WEBRTC_FRAME_DURATION

    print(f"\n{'=' * 45}")
    print(f"Recording: {path.name}")
    print(f"Original duration: {len(audio) / SAMPLE_RATE:.2f}s")
    print(f"WebRTC processed: {webrtc_duration:.2f}s")
    print(f"Silero processed: {silero_duration:.2f}s")
    print(
        f"WebRTC speech: "
        f"{webrtc_speech / webrtc_duration * 100:.2f}%"
    )
    print(
        f"Silero speech: "
        f"{silero_speech / silero_duration * 100:.2f}%"
    )


def evaluate_controlled() -> None:
    audio = load_audio(CONTROLLED_FILE)

    silero_results = silero.detect(audio, SAMPLE_RATE)
    webrtc_results = webrtc.detect(audio, SAMPLE_RATE)

    webrtc_metrics = calculate_metrics(
        webrtc_results,
        WEBRTC_FRAME_DURATION,
    )

    silero_metrics = calculate_metrics(
        silero_results,
        512 / SAMPLE_RATE,
    )

    print(f"\n{'=' * 45}")
    print("CONTROLLED SPEECH EVALUATION")
    print("Ground truth: approximate manual intervals")
    print("Evaluation window: 0–30 seconds")

    print_metrics("WebRTC VAD", webrtc_metrics)
    print_metrics("Silero VAD", silero_metrics)


def main() -> None:
    if not CONTROLLED_FILE.is_file():
        raise FileNotFoundError(CONTROLLED_FILE)

    evaluate_controlled()

    for path in sorted(RECORDINGS.glob("*.wav")):
        if path == CONTROLLED_FILE:
            continue

        evaluate_environment(path)


if __name__ == "__main__":
    main()
