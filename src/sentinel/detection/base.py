from typing import Protocol

from sentinel.analysis.audio_analyzer import AudioFrameFeatures
from sentinel.detection.models import DetectionResult


class AudioDetector(Protocol):
    def detect(
        self,
        features: list[AudioFrameFeatures],
    ) -> list[DetectionResult]:
        ...
