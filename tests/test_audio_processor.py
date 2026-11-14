import numpy as np
import pytest

from sentinel.processing.audio_processor import AudioProcessor


def test_processor_measures_audio():
    audio = np.array([0.5, -0.5, 0.5, -0.5], dtype=np.float32)

    processor = AudioProcessor()
    result = processor.process(audio, sample_rate=16000)

    assert result.measurements.peak_amplitude == pytest.approx(0.5)
    assert result.measurements.rms_amplitude == pytest.approx(0.5)
    assert result.measurements.clipping_ratio == 0.0
    assert result.measurements.duration_seconds == pytest.approx(0.00025)


def test_processor_preserves_original_audio():
    audio = np.array([0.2, -0.4, 0.6], dtype=np.float32)

    result = AudioProcessor().process(audio, sample_rate=16000)

    np.testing.assert_array_equal(result.original_audio, audio)

    assert result.original_audio is not audio


def test_processor_detects_clipping():
    audio = np.array([0.2, 0.999, -1.0, 0.5], dtype=np.float32)

    result = AudioProcessor().process(audio, sample_rate=16000)

    assert result.measurements.clipping_ratio == pytest.approx(0.5)


def test_processor_rejects_empty_audio():
    with pytest.raises(ValueError):
        AudioProcessor().process(
            np.array([], dtype=np.float32),
            sample_rate=16000,
        )


def test_processor_rejects_invalid_sample_rate():
    audio = np.array([0.2, 0.4], dtype=np.float32)

    with pytest.raises(ValueError):
        AudioProcessor().process(audio, sample_rate=0)
