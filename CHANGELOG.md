# Changelog

## 2026-11-14 — Discord Webhook Error Details

### Changed
- Include Discord's HTTP error code and message when webhook delivery fails,
  while keeping the webhook URL out of the log.

### Tested
- Added a mocked 403 response test confirming the Discord error is surfaced
  without exposing the configured webhook URL.
- Notification tests: 4 passed; Ruff checks passed.

## 2026-11-14 — Discord Speech Alerts

### Changed
- Replaced ntfy topic publishing with Discord incoming webhooks.
- Speech-start events post `Noise detected.` to the configured Discord channel.
- Added local `.env` loading for `SENTINEL_DISCORD_WEBHOOK_URL`; the webhook
  secret is not stored in the repository.
- Updated setup guidance for webhook creation and phone push notifications.

### Tested
- Discord webhook request construction and URL validation are covered with
  mocked HTTP responses; no message is sent to a real Discord channel.

## 2026-11-14 — ntfy Speech Alerts

### Added
- Added an ntfy publisher that sends a notification when speech starts.
- Added a separate notification worker so network requests do not block audio
  capture or VAD processing.
- Added configuration for ntfy topic, server, and optional bearer token through
  environment variables.
- Documented phone subscription and Pi configuration steps.

### Tested
- ntfy publishing request construction and disabled-without-topic behavior are
  covered with mocked HTTP responses; no message is sent to an external topic.

## 2026-11-14 — Live Speech/Silence Prototype

### Added
- Added an asynchronous live microphone monitor using `sounddevice.InputStream`,
  a bounded audio queue, a processing worker, SoXR streaming resampling, and
  Silero VAD.
- Added stable speech-start and speech-end terminal events with start
  confirmation and silence hangover.
- Added `run_sentinel.sh`, which performs first-run setup when dependencies are
  missing, and `scripts/setup_pi.sh` for Raspberry Pi OS dependencies.
- Added state-transition and end-of-stream tests.

### Notes
- The monitor does not save raw audio. Microphone callback errors or a full
  processing queue stop the stream with an explicit error.
- Raspberry Pi 5 hardware behavior and ARM64 package installation still need
  verification on the device.

### Tested
- Speech-state tests and Silero tests: 18 passed.
- Ruff checks, shell syntax, and whitespace checks passed.
- Real ONNX model processed one second of synthetic silence; maximum speech
  probability was 0.0089.
- SoXR streaming resampler converted 44.1 kHz input to 16 kHz output.

## 2026-11-14 — Stateful Silero Streaming

### Added
- Added `detect_stream()` to preserve Silero recurrent state and 64-sample
  context across audio blocks, buffering input until complete 512-sample frames
  are available.
- Added `reset_stream()` to clear model state, buffered samples, and stream
  timestamps before an independent stream.
- Added tests covering streaming state continuity, partial-frame buffering, and
  reset behavior.

### Tested
- Silero test suite: 14 passed.
- Ruff checks passed for the repository.
- Full-suite collection is currently blocked by missing `sounddevice` and
  `webrtcvad` packages in the test environment.

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
