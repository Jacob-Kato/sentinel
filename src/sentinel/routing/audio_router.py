from dataclasses import dataclass


@dataclass(frozen=True)
class AudioAnalysis:
    speech_confidence: float
    sound_confidence: float


@dataclass(frozen=True)
class AnalysisRoutes:
    speech_analysis: bool
    sound_analysis: bool


class AudioRouter:
    def __init__(
        self,
        speech_threshold: float = 0.60,
        sound_threshold: float = 0.60,
    ):
        if not 0.0 <= speech_threshold <= 1.0:
            raise ValueError("Speech threshold must be between 0 and 1.")

        if not 0.0 <= sound_threshold <= 1.0:
            raise ValueError("Sound threshold must be between 0 and 1.")

        self.speech_threshold = speech_threshold
        self.sound_threshold = sound_threshold

    def route(self, analysis: AudioAnalysis) -> AnalysisRoutes:
        if not 0.0 <= analysis.speech_confidence <= 1.0:
            raise ValueError("Speech confidence must be between 0 and 1.")

        if not 0.0 <= analysis.sound_confidence <= 1.0:
            raise ValueError("Sound confidence must be between 0 and 1.")

        speech_detected = (
            analysis.speech_confidence >= self.speech_threshold
        )

        sound_detected = (
            analysis.sound_confidence >= self.sound_threshold
        )

        # If neither specialized path has sufficient confidence,
        # use general sound analysis rather than discard the audio.
        if not speech_detected and not sound_detected:
            sound_detected = True

        return AnalysisRoutes(
            speech_analysis=speech_detected,
            sound_analysis=sound_detected,
        )
