from sentinel.detection.silero_vad import SileroVADFrame
from sentinel.live import SpeechStateTracker


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
