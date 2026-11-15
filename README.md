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

## Live Speech Check

On Raspberry Pi OS 64-bit, clone this repository and start the prototype with:

```bash
git clone https://github.com/Jacob-Kato/sentinel.git
cd sentinel
./run_sentinel.sh
```

On first run, the script installs PortAudio support, creates `.venv`, and
downloads the Python dependencies, including ONNX Runtime and the streaming
resampler. It then starts listening. Press `Ctrl+C` to stop. The script uses
the operating system's default input microphone and does not save raw audio.

To select a particular input device, set `SENTINEL_DEVICE` to its PortAudio
index or name before starting. The capture rate defaults to the device's
reported default and is resampled to 16 kHz for Silero VAD.

To see available input devices if the wrong microphone is selected:

```bash
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'
SENTINEL_DEVICE=1 ./run_sentinel.sh
```

Speak near the microphone. The terminal should print `Speech detected`, then
`Speech ended` after a short quiet pause. Press `Ctrl+C` to stop listening.

### Send speech alerts to ntfy

ntfy works by having the phone subscribe to a topic while Sentinel posts a
message to that same topic. Install the ntfy app on the phone, add a topic, and
use the exact same topic on the Pi. On the Pi:

```bash
export SENTINEL_NTFY_TOPIC="sentinel-$(python3 -c 'import secrets; print(secrets.token_hex(16))')"
echo "$SENTINEL_NTFY_TOPIC"
```

Subscribe to the printed topic in the phone app, then start Sentinel in that
same Pi terminal:

```bash
./run_sentinel.sh
```

The default server is `https://ntfy.sh`. You can set `SENTINEL_NTFY_SERVER` to
use a different ntfy server, or `SENTINEL_NTFY_TOKEN` if that server requires
an access token. A speech-start event sends one `Speech detected` notification;
speech-end events stay in the terminal. The notification contains no audio.

Topics on the public ntfy.sh service are public to anyone who knows the topic,
so choose a long, hard-to-guess name and keep it private. If you need access
controls, use an authenticated ntfy server and set its token on the Pi.

## Project Status

Early development. The live speech check continuously captures audio and
reports speech start/end transitions. Selected phrase recognition, emergency
sound classification, and phone notifications are not implemented yet.
