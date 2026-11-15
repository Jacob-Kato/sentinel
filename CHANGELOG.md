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
