# Sentinel

Sentinel is a situational-awareness platform that connects information
from sensors and human reports to create a continuously updated
understanding of an environment.

## Current Development Focus

Personal Sentinel — an audio-based environmental awareness prototype.

### Initial Capabilities

- Capture audio using a Raspberry Pi microphone.
- Classify speech using WebRTC VAD or ONNX-based Silero VAD.
- Detect selected spoken phrases.
- Identify possible emergency sounds.
- Convert detections into structured events.
- Send event notifications to an iPhone.
- Process audio locally without retaining raw recordings.

## Architecture

- Language: Python
- Development environment: Ubuntu
- Hardware target: Raspberry Pi 5
- Notification target: iPhone

## Development Workflow

1. Develop and test on Ubuntu.
2. Commit changes to Git.
3. Push changes to GitHub.
4. Deploy to Raspberry Pi.
5. Test against physical hardware.

## Project Status

Early development. Audio capture and batch voice activity detection are
implemented. Silero VAD also exposes a stateful streaming interface that keeps
model state across blocks; continuous microphone capture and speech event
output are not yet wired together.
