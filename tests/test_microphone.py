import numpy as np

from sentinel.sensors.microphone import Microphone


def test_microphone_configuration():
    microphone = Microphone()

    assert microphone.sample_rate == 44100
    assert microphone.channels == 1
    assert microphone.chunk_size == 44100


def test_microphone_capture():
    microphone = Microphone()

    audio = microphone.capture()

    assert isinstance(audio, np.ndarray)
    assert len(audio) == 44100
    assert audio.dtype == np.float32
