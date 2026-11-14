from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class AudioFrameFeatures:
    timestamp_seconds: float
    rms_energy: float
    zero_crossing_rate: float
    spectral_centroid: float
    spectral_flatness: float


class AudioAnalyzer:
    def __init__(
        self,
        window_duration: float = 0.025,
        hop_duration: float = 0.010,
    ):
        if window_duration <= 0 or hop_duration <= 0:
            raise ValueError("Window and hop durations must be positive.")

        if hop_duration > window_duration:
            raise ValueError("Hop duration cannot exceed window duration.")

        self.window_duration = window_duration
        self.hop_duration = hop_duration

    def analyze(
        self,
        audio: npt.NDArray[np.float32],
        sample_rate: int,
    ) -> list[AudioFrameFeatures]:
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive.")

        if audio.ndim != 1:
            raise ValueError("Audio must be a one-dimensional array.")

        if audio.size == 0:
            raise ValueError("Audio cannot be empty.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid values.")

        window_size = round(sample_rate * self.window_duration)
        hop_size = round(sample_rate * self.hop_duration)

        if audio.size < window_size:
            raise ValueError("Audio is shorter than one analysis window.")

        features = []

        window = np.hanning(window_size)

        for start in range(0, audio.size - window_size + 1, hop_size):
            frame = audio[start : start + window_size]

            rms_energy = float(
                np.sqrt(np.mean(frame**2))
            )

            zero_crossings = np.count_nonzero(
                np.diff(np.signbit(frame))
            )

            zero_crossing_rate = float(
                zero_crossings / (window_size - 1)
            )

            spectrum = np.abs(
                np.fft.rfft(frame * window)
            )

            frequencies = np.fft.rfftfreq(
                window_size,
                d=1.0 / sample_rate,
            )

            spectral_sum = float(np.sum(spectrum))

            if spectral_sum > 0:
                spectral_centroid = float(
                    np.sum(frequencies * spectrum) / spectral_sum
                )
            else:
                spectral_centroid = 0.0

            power_spectrum = spectrum**2
            arithmetic_mean = float(np.mean(power_spectrum))

            if arithmetic_mean > 0:
                positive_power = power_spectrum[power_spectrum > 0]

                geometric_mean = float(
                    np.exp(np.mean(np.log(positive_power)))
                )

                spectral_flatness = geometric_mean / arithmetic_mean
            else:
                spectral_flatness = 0.0

            features.append(
                AudioFrameFeatures(
                    timestamp_seconds=start / sample_rate,
                    rms_energy=rms_energy,
                    zero_crossing_rate=zero_crossing_rate,
                    spectral_centroid=spectral_centroid,
                    spectral_flatness=spectral_flatness,
                )
            )

        return features
