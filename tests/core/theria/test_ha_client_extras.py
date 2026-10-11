from datetime import UTC, datetime, timedelta

import pytest
import requests

from core.theria.ha_client import HAClient, HANotifier

START = datetime(2026, 10, 10, tzinfo=UTC)


class FakeResponse:
    def __init__(self, payload=None, status=200):
        self._payload, self.status_code = payload, status
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(response=self)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response, self.error = response, error
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        if self.error:
            raise self.error
        return self.response

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json))
        if self.error:
            raise self.error
        return self.response


def client(session) -> HAClient:
    ha = HAClient("http://ha", "token")
    ha.session = session
    return ha


def test_numeric_history_parses_state_changes_and_skips_non_numeric_values():
    payload = [
        [
            {
                "entity_id": "sensor.x",
                "state": "20.5",
                "last_changed": "2026-10-10T01:00:00+00:00",
            },
            {"state": "unavailable", "last_changed": "2026-10-10T02:00:00+00:00"},
            {"state": "21.0", "last_changed": "2026-10-10T03:00:00+00:00"},
        ]
    ]
    session = FakeSession(FakeResponse(payload))
    points = client(session).get_numeric_history(
        "sensor.x", START, START + timedelta(days=1)
    )
    assert [(t.hour, v) for t, v in points] == [(1, 20.5), (3, 21.0)]
    _, url, params = session.calls[0]
    assert url.startswith("http://ha/api/history/period/2026-10-10T00:00:00")
    assert params["filter_entity_id"] == "sensor.x" and "minimal_response" in params


def test_numeric_history_of_an_entity_without_records_is_empty():
    assert (
        client(FakeSession(FakeResponse([]))).get_numeric_history(
            "sensor.x", START, START + timedelta(days=1)
        )
        == []
    )


def test_history_request_failure_is_a_runtime_error():
    ha = client(FakeSession(error=requests.exceptions.ConnectionError("down")))
    with pytest.raises(RuntimeError):
        ha.get_numeric_history("sensor.x", START, START + timedelta(days=1))


def test_notifier_creates_and_dismisses_persistent_notifications():
    session = FakeSession(FakeResponse({}))
    notifier = HANotifier(client(session))
    notifier.notify("theria_x", "Titel", "Text")
    notifier.dismiss("theria_x")
    assert session.calls[0] == (
        "POST",
        "http://ha/api/services/persistent_notification/create",
        {"notification_id": "theria_x", "title": "Titel", "message": "Text"},
    )
    assert session.calls[1] == (
        "POST",
        "http://ha/api/services/persistent_notification/dismiss",
        {"notification_id": "theria_x"},
    )


def test_notifier_failure_is_a_runtime_error_not_silent():
    notifier = HANotifier(
        client(FakeSession(error=requests.exceptions.ConnectionError("down")))
    )
    with pytest.raises(RuntimeError):
        notifier.notify("theria_x", "t", "m")
