from types import SimpleNamespace

import pytest

import sentinel.live as live
from sentinel.detection.silero_vad import SileroVADFrame
from sentinel.live import SpeechStateTracker, _select_input_device


def make_frame(timestamp, is_speech, probability=None):
    if probability is None:
        probability = 0.9 if is_speech else 0.1
    return SileroVADFrame(
        timestamp_seconds=timestamp,
        speech_probability=probability,
        is_speech=is_speech,
        duration_seconds=0.032,
    )


def test_tracker_emits_start_after_consecutive_speech_frames():
    tracker = SpeechStateTracker(start_frames=2, hangover_frames=2)

    assert tracker.update(make_frame(0.0, True)) is None
    event = tracker.update(make_frame(0.032, True))

    assert event is not None
    assert event.kind == "speech_started"
    assert event.timestamp_seconds == 0.0
    assert tracker.speaking


def test_tracker_uses_hangover_before_ending_speech():
    tracker = SpeechStateTracker(start_frames=1, hangover_frames=2)
    tracker.update(make_frame(0.0, True))

    assert tracker.update(make_frame(0.032, False)) is None
    assert tracker.speaking
    event = tracker.update(make_frame(0.064, False))

    assert event is not None
    assert event.kind == "speech_ended"
    assert event.timestamp_seconds == 0.032
    assert not tracker.speaking


def test_tracker_does_not_emit_start_for_single_positive_frame():
    tracker = SpeechStateTracker(start_frames=2, hangover_frames=2)

    assert tracker.update(make_frame(0.0, True)) is None
    assert tracker.update(make_frame(0.032, False)) is None
    assert not tracker.speaking


def test_tracker_emits_a_new_start_after_speech_ends():
    tracker = SpeechStateTracker(start_frames=1, hangover_frames=2)

    first_start = tracker.update(make_frame(0.0, True))
    assert first_start is not None
    assert first_start.kind == "speech_started"
    assert tracker.update(make_frame(0.032, False)) is None
    first_end = tracker.update(make_frame(0.064, False))
    assert first_end is not None
    assert first_end.kind == "speech_ended"

    second_start = tracker.update(make_frame(0.096, True))
    assert second_start is not None
    assert second_start.kind == "speech_started"


def test_uses_available_usb_input_when_no_default_is_selected(monkeypatch):
    devices = [
        {"name": "HDMI Output", "max_input_channels": 0},
        {
            "name": "USB PnP Sound Device",
            "max_input_channels": 1,
            "default_samplerate": 44100,
        },
    ]
    monkeypatch.delenv("SENTINEL_DEVICE", raising=False)
    monkeypatch.delenv("SENTINEL_INPUT_RATE", raising=False)
    monkeypatch.setattr(live.sd, "default", SimpleNamespace(device=(-1, -1)))

    def fake_query_devices(*args):
        if not args:
            return devices
        return devices[args[0]]

    monkeypatch.setattr(live.sd, "query_devices", fake_query_devices)
    monkeypatch.setattr(live.sd, "check_input_settings", lambda **kwargs: None)

    device, sample_rate = _select_input_device()

    assert device == 1
    assert sample_rate == 44100


def test_reports_when_no_microphone_inputs_are_found(monkeypatch):
    monkeypatch.delenv("SENTINEL_DEVICE", raising=False)
    monkeypatch.setattr(live.sd, "default", SimpleNamespace(device=(-1, -1)))
    monkeypatch.setattr(live.sd, "query_devices", lambda: [])

    with pytest.raises(RuntimeError, match="No microphone input devices"):
        _select_input_device()
