from typing import Protocol

import numpy.typing as npt

from sentinel.analysis.audio_analyzer import AudioFrameFeatures
from sentinel.routing.audio_router import AudioAnalysis


class AudioGate(Protocol):
    def analyze(
        self,
        audio: npt.NDArray,
        features: list[AudioFrameFeatures],
    ) -> AudioAnalysis:
        ...


class MockAudioGate:
    """Returns predefined confidence scores for pipeline testing."""

    def __init__(self, analysis: AudioAnalysis):
        self.analysis = analysis

    def analyze(
        self,
        audio: npt.NDArray,
        features: list[AudioFrameFeatures],
    ) -> AudioAnalysis:
        if not features:
            raise ValueError("Features cannot be empty.")

        return self.analysis
