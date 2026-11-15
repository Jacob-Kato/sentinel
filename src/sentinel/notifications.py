"""Discord webhook notifications for Sentinel events."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from sentinel.detection.alerts import AlertEvent, AlertTier


class DiscordWebhookNotifier:
    """Post short event messages to a Discord channel webhook."""

    def __init__(
        self,
        webhook_url: str | None,
        timeout_seconds: float = 5.0,
        opener: Callable = urlopen,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive.")
        self.webhook_url = webhook_url.strip() if webhook_url else ""
        self.timeout_seconds = timeout_seconds
        self._opener = opener

        if self.webhook_url:
            parsed = urlsplit(self.webhook_url)
            host = parsed.hostname or ""
            valid_host = host == "discord.com" or host.endswith(".discord.com")
            if (
                parsed.scheme != "https"
                or not valid_host
                or not parsed.path.startswith("/api/webhooks/")
            ):
                raise ValueError(
                    "Discord webhook URL must be an HTTPS Discord webhook URL."
                )

    @classmethod
    def from_environment(cls) -> DiscordWebhookNotifier:
        return cls(os.environ.get("SENTINEL_DISCORD_WEBHOOK_URL"))

    @property
    def enabled(self) -> bool:
        return bool(self.webhook_url)

    def send_alert(self, event: AlertEvent) -> bool:
        """Send one tiered notification; return False without a webhook."""
        if not self.enabled:
            return False

        parts = urlsplit(self.webhook_url)
        query = dict(parse_qsl(parts.query, keep_blank_values=True))
        query["wait"] = "true"
        url = urlunsplit(
            (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
        )
        prefix = {
            AlertTier.SPEECH: "",
            AlertTier.ATTENTION: "ATTENTION: ",
            AlertTier.EMERGENCY: "EMERGENCY: ",
        }[event.tier]
        payload = {
            "content": f"{prefix}{event.message}",
            "username": "Sentinel",
            "allowed_mentions": {"parse": []},
        }
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": (
                    "DiscordBot (https://github.com/Jacob-Kato/sentinel, 0.1.0)"
                ),
            },
            method="POST",
        )

        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                status = getattr(response, "status", 204)
                if not 200 <= status < 300:
                    raise RuntimeError(f"Discord returned HTTP {status}.")
        except HTTPError as exc:
            detail = ""
            try:
                error_body = json.loads(exc.read(4096).decode("utf-8"))
                code = error_body.get("code")
                message = error_body.get("message")
                if code is not None or message:
                    detail = f"; Discord code {code}: {message or 'unspecified error'}"
            except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
                pass
            raise RuntimeError(
                f"Discord rejected the webhook (HTTP {exc.code}{detail})."
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Discord: {exc.reason}") from exc
        return True

    def send_noise_detected(self) -> bool:
        """Keep the original one-line noise notification API available."""
        return self.send_alert(
            AlertEvent(AlertTier.NOISE, "Noise detected.", timestamp_seconds=0.0)
        )
