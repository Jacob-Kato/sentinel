import pytest
from sentinel.analysis.audio_analyzer import AudioFrameFeatures

from sentinel.detection.models import (
    DetectionCategory,
    DetectionResult,
)

from sentinel.detection.mock_detector import MockAudioDetector


def test_detection_result_creation():
    result = DetectionResult(
        category=DetectionCategory.SPEECH,
        confidence=0.95,
        timestamp_seconds=0.25,
        source="mock_detector",
    )

    assert result.category == DetectionCategory.SPEECH
    assert result.confidence == 0.95
    assert result.timestamp_seconds == 0.25


def test_detection_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        DetectionResult(
            category=DetectionCategory.ALARM,
            confidence=1.5,
            timestamp_seconds=0.25,
            source="mock_detector",
        )


def test_detection_rejects_negative_timestamp():
    with pytest.raises(ValueError):
        DetectionResult(
            category=DetectionCategory.SIREN,
            confidence=0.90,
            timestamp_seconds=-1,
            source="mock_detector",
        )


def test_mock_detector_returns_results():
    expected = DetectionResult(
        category=DetectionCategory.ALARM,
        confidence=0.92,
        timestamp_seconds=0.45,
        source="mock_detector",
    )

    detector = MockAudioDetector([expected])

    features = [
        AudioFrameFeatures(
            timestamp_seconds=0.0,
            rms_energy=0.5,
            zero_crossing_rate=0.2,
            spectral_centroid=1000.0,
            spectral_flatness=0.3,
        )
    ]

    result = detector.detect(features)

    assert len(result) == 1
    assert result[0] == expected


def test_mock_detector_returns_empty_for_no_features():
    detector = MockAudioDetector([])

    result = detector.detect([])

    assert result == []
