import numpy as np

from sentinel.analysis.audio_analyzer import AudioAnalyzer
from sentinel.analysis.audio_gate import MockAudioGate
from sentinel.detection.mock_detector import MockAudioDetector
from sentinel.detection.models import (
    DetectionCategory,
    DetectionResult,
)
from sentinel.pipeline import AudioPipeline
from sentinel.processing.audio_processor import AudioProcessor
from sentinel.routing.audio_router import AudioAnalysis, AudioRouter


class TrackingDetector(MockAudioDetector):
    def __init__(self, results):
        super().__init__(results)
        self.called = False

    def detect(self, features):
        self.called = True
        return super().detect(features)


def create_pipeline(speech_confidence, sound_confidence):
    speech_result = DetectionResult(
        category=DetectionCategory.SPEECH,
        confidence=0.95,
        timestamp_seconds=0.1,
        source="speech_detector",
    )

    sound_result = DetectionResult(
        category=DetectionCategory.ALARM,
        confidence=0.90,
        timestamp_seconds=0.2,
        source="sound_detector",
    )

    speech_detector = TrackingDetector([speech_result])
    sound_detector = TrackingDetector([sound_result])

    pipeline = AudioPipeline(
        processor=AudioProcessor(),
        analyzer=AudioAnalyzer(),
        gate=MockAudioGate(
            AudioAnalysis(
                speech_confidence=speech_confidence,
                sound_confidence=sound_confidence,
            )
        ),
        router=AudioRouter(),
        speech_detector=speech_detector,
        sound_detector=sound_detector,
    )

    return pipeline, speech_detector, sound_detector


def test_pipeline_routes_speech_only():
    pipeline, speech, sound = create_pipeline(0.95, 0.10)

    audio = np.zeros(16000, dtype=np.float32)

    result = pipeline.process(audio, 16000)

    assert speech.called is True
    assert sound.called is False
    assert len(result.detections) == 1
    assert result.detections[0].category == DetectionCategory.SPEECH


def test_pipeline_routes_sound_only():
    pipeline, speech, sound = create_pipeline(0.10, 0.95)

    audio = np.zeros(16000, dtype=np.float32)

    result = pipeline.process(audio, 16000)

    assert speech.called is False
    assert sound.called is True
    assert len(result.detections) == 1
    assert result.detections[0].category == DetectionCategory.ALARM


def test_pipeline_routes_both():
    pipeline, speech, sound = create_pipeline(0.95, 0.95)

    audio = np.zeros(16000, dtype=np.float32)

    result = pipeline.process(audio, 16000)

    assert speech.called is True
    assert sound.called is True
    assert len(result.detections) == 2


def test_pipeline_routes_uncertain_audio_to_sound():
    pipeline, speech, sound = create_pipeline(0.20, 0.30)

    audio = np.zeros(16000, dtype=np.float32)

    result = pipeline.process(audio, 16000)

    assert speech.called is False
    assert sound.called is True
    assert len(result.detections) == 1


def test_pipeline_returns_processing_and_features():
    pipeline, _, _ = create_pipeline(0.95, 0.95)

    audio = np.zeros(16000, dtype=np.float32)

    result = pipeline.process(audio, 16000)

    assert result.processing.measurements.sample_count == 16000
    assert result.processing.measurements.duration_seconds == 1.0
    assert len(result.features) > 0
    assert result.routes.speech_analysis is True
    assert result.routes.sound_analysis is True
