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

### Send speech alerts to Discord

Create a text channel in a Discord server for Sentinel alerts. In Discord
desktop or web, open **Server Settings → Integrations → Webhooks**, create a
webhook for that channel, and copy its URL. Discord webhooks post messages into
the chosen channel. The webhook URL is a secret: do not commit it or share it
in chat.

On the Pi, create a local `.env` file in the repository:

```bash
cat > .env <<'EOF'
SENTINEL_DISCORD_WEBHOOK_URL="paste-your-discord-webhook-url-here"
EOF
chmod 600 .env
```

Replace the placeholder with the copied URL, then run `./run_sentinel.sh`.
The launcher reads `.env`; `.env` is ignored by Git. A speech-start event
posts `Noise detected.` to Discord. Speech-end events stay in the terminal,
and the notification does not include audio.

On your phone, install/open Discord and enable notifications for the server and
the Sentinel channel. Set that channel to notify for **All Messages** and make
sure **Mobile Push Notifications** and the Discord notifications permission
in your phone's system settings are enabled. If the server is set to mentions
only, the webhook message will not alert you because it does not mention you.

### Notification cadence and troubleshooting

Sentinel sends one alert at the start of each speech segment, rather than an
alert for every 32 ms VAD frame. It waits for 8 consecutive non-speech frames
(about 256 ms of quiet) before ending a segment, so another alert is sent only
after that quiet gap and a new speech start. This prevents alerts from
repeating rapidly while someone keeps speaking.

Discord's Android app can group several messages from one channel into one
notification; its **Notify on every new message in conversations** setting
controls whether later messages also make a sound or buzz. Discord also does
not send a push while the app is open and in focus. On iOS, notification
grouping is controlled under **Settings → Notifications → Discord**. See
Discord's [mobile notification guide](https://support.discord.com/hc/en-us/articles/218892547--Mobile-Notifications-Settings-101).
In the Sentinel terminal, each new event should produce both `Speech detected` and
`Discord notification sent`. If those lines repeat but the phone only alerts
once, check Discord and phone notification settings. If `Speech detected`
doesn't repeat, leave at least a quarter-second of quiet between speech
segments.

## Project Status

Early development. The live speech check continuously captures audio and
reports speech start/end transitions. Selected phrase recognition, emergency
sound classification, and phone notifications are not implemented yet.
