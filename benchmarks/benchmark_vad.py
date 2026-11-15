import time

import numpy as np

from sentinel.detection.webrtc_vad import WebRTCVAD

SAMPLE_RATE = 16000
AUDIO_DURATION = 9
ITERATIONS = 100

rng = np.random.default_rng(42)

audio = rng.normal(
    0,
    0.05,
    SAMPLE_RATE * AUDIO_DURATION,
).astype(np.float32)

print("WEBRTC VAD FRAME DURATION BENCHMARK")
print("-" * 55)

for frame_duration in (10, 20, 30):
    vad = WebRTCVAD(
        aggressiveness=2,
        frame_duration_ms=frame_duration,
    )

    vad.detect(audio, SAMPLE_RATE)

    start = time.perf_counter()

    for _ in range(ITERATIONS):
        results = vad.detect(audio, SAMPLE_RATE)

    elapsed = time.perf_counter() - start

    total_audio = AUDIO_DURATION * ITERATIONS
    rtf = elapsed / total_audio

    print(f"\nFrame duration: {frame_duration} ms")
    print(f"Frames per iteration: {len(results)}")
    print(f"Total processing time: {elapsed:.4f}s")
    print(f"Average processing time: {elapsed / ITERATIONS:.6f}s")
    print(f"Real-time factor: {rtf:.8f}")
    print(f"Processing speed: {1 / rtf:.2f}x")
