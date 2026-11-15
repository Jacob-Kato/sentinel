# Audio Hardware Configuration

## Development Environment

- Operating System: Ubuntu Linux
- Language: Python 3.14.4
- Audio library: SoundDevice
- Native audio backend: PortAudio
- Hardware: USB PnP Sound Device (C-Media)

## Microphone Specifications

| Property | Value |
|---|---|
| Device | USB PnP Sound Device |
| ALSA Device | hw:2,0 |
| Channels | 1 |
| Format | S16_LE |
| Supported sample rates | 44,100 Hz and 48,000 Hz |

## Implementation Decision

The initial implementation requested 16,000 Hz, which the microphone did not support natively.

PortAudio returned:

`Invalid sample rate [PaErrorCode -9997]`

The capture configuration was changed to 44,100 Hz.

## Current Audio Pipeline

USB Microphone → SoundDevice InputStream → bounded audio queue → SoXR
streaming resampler → Silero VAD at 16,000 Hz

The capture callback only copies microphone blocks into the queue. A separate
worker performs resampling and VAD inference. Speech start/end transitions are
printed to the terminal. If a Discord webhook URL is configured, a separate
worker also sends a notification when speech starts. Raw audio is held in
memory only and is not written to disk.

One notification is sent per speech segment, not per VAD frame. The speech
tracker requires 8 consecutive non-speech frames (about 256 ms) to end a
segment, which prevents repeated alerts during continuous speech. Discord
mobile may group several channel messages into one push notification; the
Sentinel terminal logs each successful webhook send separately.

No audio recordings are written to disk by the microphone module.

## Future Improvements

- Verify capture rates and default device behavior on Raspberry Pi 5.
- Measure CPU, memory, inference latency, and queue stability during extended
  listening on Raspberry Pi 5.
- Add hardware-independent microphone tests.
- Investigate NumPy deprecation warnings.
