"""Alert levels and event payloads used by the live monitor."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class AlertTier(IntEnum):
    NOISE = 1
    ATTENTION = 2
    EMERGENCY = 3


@dataclass(frozen=True)
class AlertEvent:
    tier: AlertTier
    message: str
    timestamp_seconds: float
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("Alert message cannot be empty.")
        if self.timestamp_seconds < 0:
            raise ValueError("Alert timestamp cannot be negative.")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Alert confidence must be between 0 and 1.")
