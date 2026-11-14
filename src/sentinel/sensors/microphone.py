import sounddevice as sd
import numpy as np


class Microphone:
    def __init__(
        self,
        device: int = 6,
        sample_rate: int = 44100,
        channels: int = 1,
        chunk_duration: float = 1.0,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration = chunk_duration

        self.chunk_size = int(sample_rate * chunk_duration)

    def capture(self) -> np.ndarray:
        """Capture one audio chunk and return its samples."""

        audio = sd.rec(
            frames=self.chunk_size,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            device=self.device,
        )

        sd.wait()

        return audio.flatten()
