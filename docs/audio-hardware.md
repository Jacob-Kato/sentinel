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

USB Microphone → SoundDevice → NumPy Audio Array

Audio is captured in one-second chunks and returned as float32 samples.

No audio recordings are written to disk by the microphone module.

## Future Improvements

- Implement continuous audio streaming.
- Add audio preprocessing.
- Resample captured audio to 16,000 Hz for compatible detection models.
- Add hardware-independent microphone tests.
- Investigate NumPy deprecation warnings.
