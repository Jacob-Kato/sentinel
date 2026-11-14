import pytest

from sentinel.routing.audio_router import AudioAnalysis, AudioRouter


def test_routes_to_speech_analysis():
    router = AudioRouter()

    result = router.route(
        AudioAnalysis(
            speech_confidence=0.95,
            sound_confidence=0.10,
        )
    )

    assert result.speech_analysis is True
    assert result.sound_analysis is False


def test_routes_to_sound_analysis():
    router = AudioRouter()

    result = router.route(
        AudioAnalysis(
            speech_confidence=0.10,
            sound_confidence=0.95,
        )
    )

    assert result.speech_analysis is False
    assert result.sound_analysis is True


def test_routes_to_both_analysis_paths():
    router = AudioRouter()

    result = router.route(
        AudioAnalysis(
            speech_confidence=0.90,
            sound_confidence=0.90,
        )
    )

    assert result.speech_analysis is True
    assert result.sound_analysis is True


def test_uncertain_audio_uses_general_sound_analysis():
    router = AudioRouter()

    result = router.route(
        AudioAnalysis(
            speech_confidence=0.20,
            sound_confidence=0.30,
        )
    )

    assert result.speech_analysis is False
    assert result.sound_analysis is True


def test_rejects_invalid_confidence():
    router = AudioRouter()

    with pytest.raises(ValueError):
        router.route(
            AudioAnalysis(
                speech_confidence=1.5,
                sound_confidence=0.5,
            )
        )
