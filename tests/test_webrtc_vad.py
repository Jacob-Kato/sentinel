import numpy as np
import pytest

from sentinel.detection.webrtc_vad import WebRTCVAD


def test_vad_processes_complete_frames():
    vad = WebRTCVAD(frame_duration_ms=30)

    audio = np.zeros(480 * 10, dtype=np.float32)

    results = vad.detect(audio, 16000)

    assert len(results) == 10
    assert all(isinstance(result.is_speech, bool) for result in results)


def test_vad_returns_correct_timestamps():
    vad = WebRTCVAD(frame_duration_ms=30)

    audio = np.zeros(480 * 3, dtype=np.float32)

    results = vad.detect(audio, 16000)

    assert results[0].timestamp_seconds == 0.0
    assert results[1].timestamp_seconds == 0.03
    assert results[2].timestamp_seconds == 0.06


def test_vad_rejects_unsupported_sample_rate():
    vad = WebRTCVAD()

    audio = np.zeros(480, dtype=np.float32)

    with pytest.raises(ValueError):
        vad.detect(audio, 44100)


def test_vad_rejects_incomplete_frames():
    vad = WebRTCVAD()

    audio = np.zeros(500, dtype=np.float32)

    with pytest.raises(ValueError):
        vad.detect(audio, 16000)


def test_vad_rejects_invalid_aggressiveness():
    with pytest.raises(ValueError):
        WebRTCVAD(aggressiveness=4)


def test_vad_rejects_empty_audio():
    vad = WebRTCVAD()

    with pytest.raises(ValueError):
        vad.detect(np.array([], dtype=np.float32), 16000)


def test_vad_rejects_out_of_range_audio():
    vad = WebRTCVAD()

    audio = np.full(480, 1.5, dtype=np.float32)

    with pytest.raises(ValueError):
        vad.detect(audio, 16000)
def test_vad_rejects_integer_audio():
    vad = WebRTCVAD()
    audio = np.zeros(480, dtype=np.int16)

    with pytest.raises(ValueError):
        vad.detect(audio, 16000)


def test_vad_rejects_invalid_sample_rate_type():
    vad = WebRTCVAD()
    audio = np.zeros(480, dtype=np.float32)

    with pytest.raises(ValueError):
        vad.detect(audio, 16000.0)
