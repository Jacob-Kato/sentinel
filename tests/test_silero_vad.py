import numpy as np
import pytest

from sentinel.detection.silero_vad import SileroVAD


class FakeONNXSession:
    def __init__(self, probability=0.8):
        self.probability = probability
        self.calls = []

    def run(self, output_names, inputs):
        self.calls.append(inputs)

        output = np.array([[self.probability]], dtype=np.float32)
        state = inputs["state"] + 1

        return [output, state]


@pytest.fixture
def audio():
    return np.zeros(16000, dtype=np.float32)


def test_detect_processes_one_second(audio):
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    results = vad.detect(audio, 16000)

    assert len(results) == 32
    assert all(result.is_speech for result in results)


def test_detect_returns_correct_timestamps(audio):
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    results = vad.detect(audio, 16000)

    assert results[0].timestamp_seconds == 0
    assert results[1].timestamp_seconds == pytest.approx(0.032)
    assert results[2].timestamp_seconds == pytest.approx(0.064)


def test_detect_returns_speech_probability(audio):
    session = FakeONNXSession(probability=0.8)
    vad = SileroVAD(session=session)

    results = vad.detect(audio, 16000)

    assert results[0].speech_probability == pytest.approx(0.8)


def test_threshold_controls_classification(audio):
    session = FakeONNXSession(probability=0.6)

    vad = SileroVAD(threshold=0.7, session=session)
    results = vad.detect(audio, 16000)

    assert not results[0].is_speech


def test_final_partial_chunk_is_processed():
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    audio = np.zeros(768, dtype=np.float32)
    results = vad.detect(audio, 16000)

    assert len(results) == 2
    assert results[-1].duration_seconds == pytest.approx(256 / 16000)


def test_model_receives_correct_input_shape(audio):
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    vad.detect(audio, 16000)

    inputs = session.calls[0]

    assert inputs["input"].shape == (1, 576)
    assert inputs["state"].shape == (2, 1, 128)
    assert inputs["sr"].item() == 16000


def test_state_resets_between_recordings(audio):
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    vad.detect(audio, 16000)
    vad.detect(audio, 16000)

    first_state = session.calls[0]["state"]
    second_recording_state = session.calls[32]["state"]

    assert np.all(first_state == 0)
    assert np.all(second_recording_state == 0)


def test_detect_stream_carries_state_and_buffers_partial_frames():
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    first = vad.detect_stream(np.zeros(768, dtype=np.float32), 16000)
    second = vad.detect_stream(np.zeros(256, dtype=np.float32), 16000)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].timestamp_seconds == 0
    assert second[0].timestamp_seconds == pytest.approx(0.032)
    assert np.all(session.calls[0]["state"] == 0)
    assert np.all(session.calls[1]["state"] == 1)


def test_reset_stream_clears_state_context_and_timestamp():
    session = FakeONNXSession()
    vad = SileroVAD(session=session)

    vad.detect_stream(np.ones(512, dtype=np.float32), 16000)
    vad.reset_stream()
    results = vad.detect_stream(np.zeros(512, dtype=np.float32), 16000)

    assert results[0].timestamp_seconds == 0
    assert np.all(session.calls[1]["state"] == 0)
    assert np.all(session.calls[1]["input"][:, :64] == 0)


def test_rejects_unsupported_sample_rate(audio):
    vad = SileroVAD(session=FakeONNXSession())

    with pytest.raises(ValueError):
        vad.detect(audio, 44100)


def test_rejects_stereo_audio():
    vad = SileroVAD(session=FakeONNXSession())
    audio = np.zeros((16000, 2), dtype=np.float32)

    with pytest.raises(ValueError):
        vad.detect(audio, 16000)


def test_rejects_empty_audio():
    vad = SileroVAD(session=FakeONNXSession())

    with pytest.raises(ValueError):
        vad.detect(np.array([], dtype=np.float32), 16000)


def test_rejects_audio_outside_valid_range():
    vad = SileroVAD(session=FakeONNXSession())
    audio = np.array([0.0, 1.5], dtype=np.float32)

    with pytest.raises(ValueError):
        vad.detect(audio, 16000)


def test_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        SileroVAD(threshold=1.5, session=FakeONNXSession())
