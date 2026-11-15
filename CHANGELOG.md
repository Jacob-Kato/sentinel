# Changelog

## 2026-11-14

### Added
- Implemented WebRTC voice activity detection.
- Implemented Silero voice activity detection using ONNX Runtime.
- Added configurable Silero speech-probability threshold.
- Added handling for incomplete Silero audio chunks.
- Added automated tests for both VAD implementations.
- Added VAD benchmarking and evaluation scripts.
- Added the Silero ONNX model.

### Tested
- Full test suite: 49 tests passed.
- Silero model loading and inference.
- WebRTC and Silero evaluation on a 78-second speech recording.
- Silero thresholds of 0.3, 0.5, and 0.7.
## 2026-11-14 — VAD Evaluation

### Added
- Evaluated WebRTC VAD and Silero VAD using controlled speech and environmental recordings.
- Added ground-truth metric calculations to `benchmarks/evaluate_vad.py`.
- Evaluated precision, recall, F1 score, and false-positive rate.

### Results

Controlled speech recording:

| Metric | WebRTC | Silero |
|---|---:|---:|
| Precision | 89.40% | 89.61% |
| Recall | 77.60% | 73.72% |
| F1 score | 83.08% | 80.89% |
| False-positive rate | 9.20% | 8.53% |

Environmental recordings:
- WebRTC classified substantially more music as speech.
- Silero rejected nearly all tested music and room noise.
- Both models detected speech activity in television audio.

### Findings
- WebRTC demonstrated slightly higher recall and F1 on the controlled recording.
- Silero demonstrated slightly better precision and false-positive rejection.
- Silero appears more selective when processing music.
- Neither model has been independently validated against a sufficiently large labeled dataset.

### Decisions
- Retain both VAD implementations.
- Keep Silero's threshold at 0.5 provisionally.
- Keep WebRTC aggressiveness at 2 provisionally.
- Defer further parameter optimization until the live prototype is operational.

### Limitations
- Controlled recording intervals were manually estimated.
- Environmental recordings were not fully annotated.
- Results are preliminary and should not be interpreted as formal model accuracy.
