"""ntfy publishing for Sentinel events."""

from __future__ import annotations

import os
from collections.abc import Callable
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class NtfyNotifier:
    """Send short event notifications to an ntfy topic over HTTPS."""

    def __init__(
        self,
        topic: str | None,
        server: str = "https://ntfy.sh",
        token: str | None = None,
        timeout_seconds: float = 5.0,
        opener: Callable = urlopen,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive.")
        self.topic = topic.strip() if topic else ""
        self.server = server.rstrip("/")
        self.token = token.strip() if token else ""
        self.timeout_seconds = timeout_seconds
        self._opener = opener

        if self.topic and not self.server.startswith(("https://", "http://")):
            raise ValueError("ntfy server must start with http:// or https://.")
        if not self.server:
            raise ValueError("ntfy server cannot be empty.")

    @classmethod
    def from_environment(cls) -> NtfyNotifier:
        return cls(
            topic=os.environ.get("SENTINEL_NTFY_TOPIC"),
            server=os.environ.get("SENTINEL_NTFY_SERVER", "https://ntfy.sh"),
            token=os.environ.get("SENTINEL_NTFY_TOKEN"),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.topic)

    def send_speech_detected(self) -> bool:
        """Publish a speech-start alert; return False when no topic is set."""
        if not self.enabled:
            return False

        topic_path = quote(self.topic, safe="")
        request = Request(
            f"{self.server}/{topic_path}",
            data=b"Speech activity detected by Sentinel.",
            headers={
                "Content-Type": "text/plain; charset=utf-8",
                "Title": "Sentinel: Speech detected",
            },
            method="POST",
        )
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")

        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if not 200 <= status < 300:
                    raise RuntimeError(f"ntfy returned HTTP {status}.")
        except URLError as exc:
            raise RuntimeError(f"Could not reach ntfy: {exc}") from exc
        return True
