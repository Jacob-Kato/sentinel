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

1. **Noise:** A sound above the configurable microphone-level threshold for
   three consecutive 512-sample frames (about 96 ms) sends `Noise detected.`
   The event ends after about 768 ms below the threshold. This catches general
   nearby sound by amplitude; it does not name the sound.
2. **Attention:** The offline phrase recognizer listens for **“hey”** or
   **“excuse me.”** It sends an `ATTENTION` Discord message when a phrase is
   recognized. The recognizer is constrained to the configured phrases and
   keeps the audio on the Pi.
3. **Emergency:** Saying **“help,” “help me,” “emergency,” “fire alarm,”** or
   **“smoke alarm”** sends an `EMERGENCY` message. In parallel, YAMNet checks
   overlapping audio windows for **smoke detector alarm, fire alarm, siren,**
   and **civil defense siren** sounds. The sound classifier waits for repeated
   positive windows before alerting, which reduces one-off false alarms.

The first tier uses an RMS amplitude threshold, not an acoustic event model.
If quiet sounds are missed or background hum causes repeated alerts, tune the
threshold in `.env` (for example, `SENTINEL_NOISE_RMS_THRESHOLD=0.02`; increase
it to reduce sensitivity, decrease it to increase sensitivity). `0.015` is the
default.

YAMNet analyzes approximately 975 ms at a time with overlapping windows. An
emergency sound alert therefore takes around 1.5 seconds or more after the
sound begins. Its score is a model confidence, not a probability that a fire
exists. Sentinel recognizes the *sound* of an alarm; it does not detect smoke,
carbon monoxide, or fire directly. Keep certified smoke and CO alarms in place
and do not rely on Sentinel for life-safety monitoring.

Discord webhooks do not provide a notification-priority setting. Sentinel
marks higher-tier message text with `ATTENTION:` or `EMERGENCY:`; phone push
behavior still follows your Discord channel and phone notification settings.

The tiers separate broad sound activity from intentional attention phrases and
possible emergencies: ordinary sound creates a low-level event, a call for
attention gets a more direct label, and help/alarm detections are marked
urgent. This helps avoid treating every sound as an emergency while keeping
the alert type visible in Discord.

For a quick check, tap or clap near the microphone for the noise tier, say
“hey” or “excuse me” for attention, and say “help” for the emergency phrase.
To check the sound classifier, play a smoke alarm or siren recording from a
second device near Sentinel's microphone. The same sound can first produce a
noise alert and then an escalated attention or emergency alert as the other
workers classify it.

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
# Optional: increase to reduce noise alerts; decrease to catch quieter sounds.
SENTINEL_NOISE_RMS_THRESHOLD="0.015"
# Optional: decrease to make alarm/siren classification more sensitive.
SENTINEL_ALARM_THRESHOLD="0.25"
EOF
chmod 600 .env
```

Replace the placeholder with the copied URL, then run `./run_sentinel.sh`.
The launcher reads `.env`; `.env` is ignored by Git. Each detected sound event
posts a noise-tier alert. Recognized attention phrases and emergency events
post messages with their tier in the text. Messages do not include audio.

On your phone, install/open Discord and enable notifications for the server and
the Sentinel channel. Set that channel to notify for **All Messages** and make
sure **Mobile Push Notifications** and the Discord notifications permission
in your phone's system settings are enabled. If the server is set to mentions
only, the webhook message will not alert you because it does not mention you.

### Notification cadence and troubleshooting

Sentinel sends one noise-tier alert when the input rises above the sound
threshold and rearms after about 768 ms of quiet. Recognized phrases are
reported once per speech utterance and alarm-classifier alerts are latched
until the sound has been absent for several analysis windows. This avoids
posting an alert for every microphone frame or every repeated alarm pulse.

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

The live Raspberry Pi prototype supports three local alert tiers: amplitude
based sound activity, selected spoken attention/emergency phrases, and
probabilistic sound classification for common alarm and siren classes. Discord
notifications are supported. Detection thresholds and real-world performance
still need tuning with the intended microphone and room; this prototype is not
a certified alarm or life-safety device.
