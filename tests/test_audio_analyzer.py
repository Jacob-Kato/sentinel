import numpy as np
import pytest

from sentinel.analysis.audio_analyzer import AudioAnalyzer


def test_analyzer_returns_multiple_frames():
    sample_rate = 16000

    audio = np.zeros(sample_rate, dtype=np.float32)

    analyzer = AudioAnalyzer()
    features = analyzer.analyze(audio, sample_rate)

    assert len(features) > 1


def test_analyzer_detects_silence():
    sample_rate = 16000

    audio = np.zeros(4000, dtype=np.float32)

    features = AudioAnalyzer().analyze(audio, sample_rate)

    for frame in features:
        assert frame.rms_energy == 0.0
        assert frame.spectral_centroid == 0.0
        assert frame.spectral_flatness == 0.0


def test_analyzer_extracts_features_from_sine_wave():
    sample_rate = 16000
    duration = 1.0
    frequency = 440

    time = np.arange(
        int(sample_rate * duration)
    ) / sample_rate

    audio = np.sin(
        2 * np.pi * frequency * time
    ).astype(np.float32)

    features = AudioAnalyzer().analyze(audio, sample_rate)

    assert len(features) > 1

    assert all(frame.rms_energy > 0 for frame in features)

    assert all(
        frame.spectral_centroid > 0
        for frame in features
    )


def test_analyzer_rejects_short_audio():
    audio = np.zeros(100, dtype=np.float32)

    with pytest.raises(ValueError):
        AudioAnalyzer().analyze(audio, sample_rate=16000)


def test_analyzer_rejects_invalid_sample_rate():
    audio = np.zeros(1000, dtype=np.float32)

    with pytest.raises(ValueError):
        AudioAnalyzer().analyze(audio, sample_rate=0)
