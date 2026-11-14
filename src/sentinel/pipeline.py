from dataclasses import dataclass

import numpy.typing as npt

from sentinel.analysis.audio_analyzer import (
    AudioAnalyzer,
    AudioFrameFeatures,
)
from sentinel.analysis.audio_gate import AudioGate
from sentinel.detection.base import AudioDetector
from sentinel.detection.models import DetectionResult
from sentinel.processing.audio_processor import (
    AudioProcessingResult,
    AudioProcessor,
)
from sentinel.routing.audio_router import (
    AnalysisRoutes,
    AudioRouter,
)


@dataclass(frozen=True)
class AudioPipelineResult:
    processing: AudioProcessingResult
    features: list[AudioFrameFeatures]
    routes: AnalysisRoutes
    detections: list[DetectionResult]


class AudioPipeline:
    def __init__(
        self,
        processor: AudioProcessor,
        analyzer: AudioAnalyzer,
        gate: AudioGate,
        router: AudioRouter,
        speech_detector: AudioDetector,
        sound_detector: AudioDetector,
    ):
        self.processor = processor
        self.analyzer = analyzer
        self.gate = gate
        self.router = router
        self.speech_detector = speech_detector
        self.sound_detector = sound_detector

    def process(
        self,
        audio: npt.NDArray,
        sample_rate: int,
    ) -> AudioPipelineResult:
        processing = self.processor.process(audio, sample_rate)

        features = self.analyzer.analyze(
            processing.original_audio,
            sample_rate,
        )

        analysis = self.gate.analyze(
            processing.original_audio,
            features,
        )

        routes = self.router.route(analysis)

        detections = []

        if routes.speech_analysis:
            detections.extend(
                self.speech_detector.detect(features)
            )

        if routes.sound_analysis:
            detections.extend(
                self.sound_detector.detect(features)
            )

        return AudioPipelineResult(
            processing=processing,
            features=features,
            routes=routes,
            detections=detections,
        )
