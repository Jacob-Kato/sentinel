"""Live microphone speech/silence monitor."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import soxr

from sentinel.detection.silero_vad import SileroVAD, SileroVADFrame
from sentinel.notifications import NtfyNotifier


@dataclass(frozen=True)
class SpeechEvent:
    kind: str
    timestamp_seconds: float
    speech_probability: float


class SpeechStateTracker:
    """Turn frame classifications into stable speech start/end transitions."""

    def __init__(self, start_frames: int = 2, hangover_frames: int = 8) -> None:
        if start_frames < 1 or hangover_frames < 1:
            raise ValueError("Speech transition frame counts must be positive.")
        self.start_frames = start_frames
        self.hangover_frames = hangover_frames
        self.speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._candidate_start = 0.0
        self._last_speech_end = 0.0
        self._last_probability = 0.0

    def update(self, frame: SileroVADFrame) -> SpeechEvent | None:
        if frame.is_speech:
            self._last_speech_end = frame.timestamp_seconds + frame.duration_seconds
            self._last_probability = frame.speech_probability
            self._silence_run = 0
            if self.speaking:
                return None
            if self._speech_run == 0:
                self._candidate_start = frame.timestamp_seconds
            self._speech_run += 1
            if self._speech_run >= self.start_frames:
                self.speaking = True
                self._speech_run = 0
                return SpeechEvent(
                    "speech_started",
                    self._candidate_start,
                    frame.speech_probability,
                )
            return None

        self._speech_run = 0
        if not self.speaking:
            return None
        self._silence_run += 1
        if self._silence_run >= self.hangover_frames:
            self.speaking = False
            self._silence_run = 0
            return SpeechEvent(
                "speech_ended",
                self._last_speech_end,
                self._last_probability,
            )
        return None

    def finish(self) -> SpeechEvent | None:
        """Close an active speech event when the audio stream stops."""
        if not self.speaking:
            return None
        self.speaking = False
        self._silence_run = 0
        return SpeechEvent(
            "speech_ended",
            self._last_speech_end,
            self._last_probability,
        )


def _select_input_device() -> tuple[int | str | None, int]:
    configured_device = os.environ.get("SENTINEL_DEVICE")
    device: int | str | None
    if configured_device:
        device = int(configured_device) if configured_device.isdecimal() else configured_device
    else:
        device = sd.default.device[0]

    info = sd.query_devices(device, "input")
    configured_rate = os.environ.get("SENTINEL_INPUT_RATE")
    sample_rate = int(configured_rate) if configured_rate else int(
        info["default_samplerate"]
    )
    if sample_rate <= 0:
        raise RuntimeError("Could not determine a valid microphone sample rate.")
    sd.check_input_settings(
        device=device,
        channels=1,
        dtype="float32",
        samplerate=sample_rate,
    )
    return device, sample_rate


def _print_event(event: SpeechEvent) -> None:
    if event.kind == "speech_started":
        print(
            "[SENTINEL] Speech detected "
            f"(probability={event.speech_probability:.2f}).",
            flush=True,
        )
    elif event.kind == "speech_ended":
        print("[SENTINEL] Speech ended.", flush=True)
        print("[SENTINEL] Listening...", flush=True)


def run() -> None:
    """Listen until Ctrl+C, processing VAD outside the audio callback."""
    device, input_rate = _select_input_device()
    device_info = sd.query_devices(device, "input")
    device_name = device_info["name"]

    audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
    notification_queue: queue.Queue[SpeechEvent] = queue.Queue(maxsize=16)
    stop_event = threading.Event()
    notification_stop = threading.Event()
    callback_error: list[str] = []
    worker_error: list[BaseException] = []
    tracker = SpeechStateTracker()
    notifier = NtfyNotifier.from_environment()
    resampler = soxr.ResampleStream(input_rate, 16000, 1, dtype="float32")
    detector = SileroVAD()
    started_at = time.monotonic()

    def fail(message: str) -> None:
        if not callback_error:
            callback_error.append(message)
        stop_event.set()

    def audio_callback(
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        del frames, time_info
        if status:
            fail(f"Microphone callback reported an audio error: {status}")
            raise sd.CallbackAbort
        try:
            audio_queue.put_nowait(indata[:, 0].copy())
        except queue.Full:
            fail("Audio processing fell behind; the bounded audio queue is full.")
            raise sd.CallbackAbort

    def enqueue_notification(event: SpeechEvent) -> None:
        if event.kind != "speech_started" or not notifier.enabled:
            return
        try:
            notification_queue.put_nowait(event)
        except queue.Full:
            print(
                "[SENTINEL] Notification queue is full; speech alert was not sent.",
                flush=True,
            )

    def send_notifications() -> None:
        while not notification_stop.is_set() or not notification_queue.empty():
            try:
                notification_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                notifier.send_speech_detected()
                print("[SENTINEL] Phone notification sent.", flush=True)
            except Exception as exc:
                print(f"[SENTINEL] Phone notification failed: {exc}", flush=True)
            finally:
                notification_queue.task_done()

    def process_audio() -> None:
        try:
            def consume(converted_audio: np.ndarray) -> None:
                converted_audio = np.asarray(
                    converted_audio,
                    dtype=np.float32,
                ).reshape(-1)
                if not converted_audio.size:
                    return
                for frame in detector.detect_stream(converted_audio, 16000):
                    event = tracker.update(frame)
                    if event is not None:
                        _print_event(event)
                        enqueue_notification(event)

            while not stop_event.is_set() or not audio_queue.empty():
                try:
                    captured = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                consume(resampler.resample_chunk(captured, last=False))
                audio_queue.task_done()
            final_audio = resampler.resample_chunk(
                np.empty(0, dtype=np.float32),
                last=True,
            )
            consume(final_audio)
            for frame in detector.flush_stream():
                event = tracker.update(frame)
                if event is not None:
                    _print_event(event)
                    enqueue_notification(event)
            event = tracker.finish()
            if event is not None:
                _print_event(event)
        except BaseException as exc:
            worker_error.append(exc)
            stop_event.set()

    worker = threading.Thread(
        target=process_audio,
        name="sentinel-audio-processor",
        daemon=True,
    )
    worker.start()
    notification_worker: threading.Thread | None = None
    if notifier.enabled:
        notification_worker = threading.Thread(
            target=send_notifications,
            name="sentinel-ntfy-publisher",
            daemon=True,
        )
        notification_worker.start()
        print("[SENTINEL] Phone notifications enabled.", flush=True)
    else:
        print(
            "[SENTINEL] Phone notifications disabled; set SENTINEL_NTFY_TOPIC to enable.",
            flush=True,
        )
    print(f"[SENTINEL] Audio stream starting: {device_name} at {input_rate} Hz.", flush=True)

    try:
        with sd.InputStream(
            device=device,
            samplerate=input_rate,
            channels=1,
            dtype="float32",
            blocksize=0,
            callback=audio_callback,
        ):
            print("[SENTINEL] Audio stream started.", flush=True)
            print("[SENTINEL] Listening... (press Ctrl+C to stop)", flush=True)
            while not stop_event.wait(0.2):
                pass
    except KeyboardInterrupt:
        print("[SENTINEL] Stopping...", flush=True)
    finally:
        stop_event.set()
        worker.join(timeout=10)
        notification_stop.set()
        if notification_worker is not None:
            notification_worker.join(timeout=10)

    if worker.is_alive():
        raise RuntimeError("Audio processing worker did not stop cleanly.")
    if notification_worker is not None and notification_worker.is_alive():
        raise RuntimeError("Notification worker did not stop cleanly.")
    if worker_error:
        raise RuntimeError("Audio processing failed.") from worker_error[0]
    if callback_error:
        raise RuntimeError(callback_error[0])
    elapsed = time.monotonic() - started_at
    print(f"[SENTINEL] Audio stream stopped after {elapsed:.1f} seconds.", flush=True)


if __name__ == "__main__":
    run()
