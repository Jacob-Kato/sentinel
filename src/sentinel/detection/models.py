from dataclasses import dataclass
from enum import Enum


class DetectionCategory(str, Enum):
    SPEECH = "speech"
    ALARM = "alarm"
    SIREN = "siren"
    IMPACT = "impact"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DetectionResult:
    category: DetectionCategory
    confidence: float
    timestamp_seconds: float
    source: str

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0 and 1.")

        if self.timestamp_seconds < 0:
            raise ValueError("Timestamp cannot be negative.")

        if not self.source:
            raise ValueError("Detection source cannot be empty.")
