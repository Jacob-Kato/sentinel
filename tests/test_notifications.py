import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from sentinel.notifications import DiscordWebhookNotifier


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_noise_alert_posts_json_to_configured_discord_webhook():
    requests = []

    def fake_opener(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    notifier = DiscordWebhookNotifier(
        webhook_url="https://discord.com/api/webhooks/123/secret-token",
        opener=fake_opener,
    )

    assert notifier.send_noise_detected()
    request, timeout = requests[0]
    assert request.full_url.endswith("?wait=true")
    assert json.loads(request.data) == {
        "content": "Noise detected.",
        "username": "Sentinel",
        "allowed_mentions": {"parse": []},
    }
    assert request.get_header("Content-type") == "application/json"
    assert timeout == 5.0


def test_notifications_are_disabled_without_a_webhook_url():
    notifier = DiscordWebhookNotifier(webhook_url=None)

    assert not notifier.enabled
    assert not notifier.send_noise_detected()


def test_rejects_non_discord_or_non_https_webhook_urls():
    with pytest.raises(ValueError):
        DiscordWebhookNotifier("http://discord.com/api/webhooks/123/token")

    with pytest.raises(ValueError):
        DiscordWebhookNotifier("https://example.com/api/webhooks/123/token")


def test_http_error_includes_discord_reason_without_webhook_url():
    webhook_url = "https://discord.com/api/webhooks/123/private-token"

    def rejected_request(request, timeout):
        raise HTTPError(
            request.full_url,
            403,
            "Forbidden",
            {},
            BytesIO(b'{"code":50027,"message":"Invalid Webhook Token"}'),
        )

    notifier = DiscordWebhookNotifier(webhook_url, opener=rejected_request)

    with pytest.raises(RuntimeError) as error:
        notifier.send_noise_detected()

    assert "50027" in str(error.value)
    assert "Invalid Webhook Token" in str(error.value)
    assert "private-token" not in str(error.value)
