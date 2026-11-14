from sentinel.analysis.audio_analyzer import AudioFrameFeatures
from sentinel.detection.models import DetectionResult


class MockAudioDetector:
    def __init__(self, results: list[DetectionResult]):
        self.results = results

    def detect(
        self,
        features: list[AudioFrameFeatures],
    ) -> list[DetectionResult]:
        if not features:
            return []

        return self.results.copy()
