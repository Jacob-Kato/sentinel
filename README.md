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

## Live Alert Monitor

On Raspberry Pi OS 64-bit, clone this repository and start the prototype with:

```bash
git clone https://github.com/Jacob-Kato/sentinel.git
cd sentinel
./run_sentinel.sh
```

On first run, the script installs PortAudio support, creates `.venv`, downloads
the Python dependencies, and downloads the small offline speech and sound
models. It then starts listening. Press `Ctrl+C` to stop. The script uses the
operating system's default input microphone and does not save raw audio.

To select a particular input device, set `SENTINEL_DEVICE` to its PortAudio
index or name before starting. The capture rate defaults to the device's
reported default and is resampled to 16 kHz for Silero VAD.

To see available input devices if the wrong microphone is selected:

```bash
.venv/bin/python -c 'import sounddevice as sd; print(sd.query_devices())'
SENTINEL_DEVICE=1 ./run_sentinel.sh
```

Speak near the microphone and use the checks below to exercise each tier.
Silero VAD, phrase recognition, sound classification, and Discord publishing
run on separate workers so model inference does not run inside the microphone
callback. Audio is processed locally in memory and is not saved.

### Three alert tiers

1. **Speech:** Silero VAD must identify speech in three consecutive frames
   (about 96 ms) before Sentinel sends `Speech detected.` A clap, bump, or other
   loud nonverbal sound does not trigger this tier. Other people talking on a
   TV or nearby can still count as speech.
2. **Attention:** The offline phrase recognizer listens for **“hey”** or
   **“excuse me.”** The sound classifier also recognizes the AudioSet **Knock**
   class as an attention event. YAMNet's [class list](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv)
   includes `Knock` as a distinct sound label. Knock alerts require a score of
   at least `0.4` in two overlapping windows. This remains probabilistic; tune
   `SENTINEL_KNOCK_THRESHOLD` upward or set `SENTINEL_DETECT_KNOCKS=0` if
   ordinary taps cause false alerts. Test with your microphone and room.
3. **Emergency:** The default spoken triggers include **help, help me, please
   help, somebody/someone help, I need help, call 911/112, emergency, fire,
   fire alarm, smoke alarm, carbon monoxide, gas leak, get out, evacuate, call
   the police, break in,** and **intruder**. Add phrases for your needs with
   `SENTINEL_EXTRA_EMERGENCY_PHRASES`, as a comma-separated list in `.env`.
   In parallel, YAMNet checks for **smoke detector alarm, fire alarm, siren,**
   and **civil defense siren** sounds. These labels are also in YAMNet's
   [class list](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv).
   It confirms a sound across multiple windows before alerting.

YAMNet analyzes about 0.96 seconds at a time, with a new overlapping window
every 0.48 seconds. Sound alerts therefore take about 1.5 seconds or more.
Its score is model confidence, not a probability that a fire exists. Sentinel
recognizes alarm *sounds*; it does not sense smoke, carbon monoxide, or fire
directly. Keep certified smoke and CO alarms in place and do not rely on
Sentinel for life-safety monitoring.

Discord webhooks do not provide a notification-priority setting. Sentinel
marks higher-tier message text with `ATTENTION:` or `EMERGENCY:`; phone push
behavior still follows your Discord channel and phone notification settings.

The tiers separate ordinary speech from explicit requests for attention and
possible emergencies. This avoids alerting on every bump or sound while keeping
the event type visible in Discord.

For a quick check, speak a sentence for the speech tier, say “hey” or “excuse
me” for attention, and say “help” for an emergency phrase. Knock several times
near the microphone to try the knock classifier. For alarm sounds, play a smoke
alarm or siren recording from a second device. One speech event can first send
the speech alert and then a higher-tier alert if a trigger phrase is detected.

### Send alerts to Discord

Create a text channel in a Discord server for Sentinel alerts. In Discord
desktop or web, open **Server Settings → Integrations → Webhooks**, create a
webhook for that channel, and copy its URL. Discord webhooks post messages into
the chosen channel. The webhook URL is a secret: do not commit it or share it
in chat.

On the Pi, create a local `.env` file in the repository:

```bash
cat > .env <<'EOF'
SENTINEL_DISCORD_WEBHOOK_URL="paste-your-discord-webhook-url-here"
# Optional: decrease to make alarm/siren classification more sensitive.
SENTINEL_ALARM_THRESHOLD="0.25"
# Optional: raise to reduce knock alerts, or set 0 to disable knock detection.
SENTINEL_KNOCK_THRESHOLD="0.4"
SENTINEL_DETECT_KNOCKS="1"
# Optional: add your own urgent phrases, separated by commas.
SENTINEL_EXTRA_EMERGENCY_PHRASES="call my neighbor, get out now"
EOF
chmod 600 .env
```

Replace the placeholder with the copied URL, then run `./run_sentinel.sh`.
The launcher reads `.env`; `.env` is ignored by Git. Each speech segment sends
one speech-tier alert; recognized attention and emergency triggers send their
own tiered messages. Messages do not include audio.

On your phone, install/open Discord and enable notifications for the server and
the Sentinel channel. Set that channel to notify for **All Messages** and make
sure **Mobile Push Notifications** and the Discord notifications permission
in your phone's system settings are enabled. If the server is set to mentions
only, the webhook message will not alert you because it does not mention you.

### Notification cadence and troubleshooting

Sentinel sends one speech-tier alert per VAD segment. It waits for 8
consecutive non-speech frames (about 256 ms of quiet) before ending a segment.
Recognized phrases are reported once per speech utterance. Knock and alarm
classifier alerts are latched until the sound has been absent for several
analysis windows.

Discord's Android app can group several messages from one channel into one
notification; its **Notify on every new message in conversations** setting
controls whether later messages also make a sound or buzz. Discord also does
not send a push while the app is open and in focus. On iOS, notification
grouping is controlled under **Settings → Notifications → Discord**. See
Discord's [mobile notification guide](https://support.discord.com/hc/en-us/articles/218892547--Mobile-Notifications-Settings-101).
In the Sentinel terminal, each detected tier is printed with a label and a
successful Discord post prints `Discord notification sent`. If those lines
repeat but the phone only alerts once, check Discord and phone notification
settings.

## Project Status

The live Raspberry Pi prototype supports three local alert tiers: VAD-filtered
speech, spoken attention/emergency phrases, and probabilistic sound
classification for knocks, alarms, and sirens. Discord notifications are
supported. Detection thresholds and real-world performance still need tuning
with the intended microphone and room; this prototype is not a certified alarm
or life-safety device.
