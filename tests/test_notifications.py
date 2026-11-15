from sentinel.notifications import NtfyNotifier


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_speech_alert_posts_message_to_configured_topic():
    requests = []

    def fake_opener(request, timeout):
        requests.append((request, timeout))
        return FakeResponse()

    notifier = NtfyNotifier(
        topic="sentinel-topic",
        token="secret-token",
        opener=fake_opener,
    )

    assert notifier.send_speech_detected()
    request, timeout = requests[0]
    assert request.full_url == "https://ntfy.sh/sentinel-topic"
    assert request.data == b"Speech activity detected by Sentinel."
    assert request.get_header("Title") == "Sentinel: Speech detected"
    assert request.get_header("Authorization") == "Bearer secret-token"
    assert timeout == 5.0


def test_notifications_are_disabled_without_a_topic():
    notifier = NtfyNotifier(topic=None)

    assert not notifier.enabled
    assert not notifier.send_speech_detected()
