from datetime import UTC, datetime, timedelta

import pytest
import requests

from core.theria.ha_client import HAClient, HANotifier, env_test_mode

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
        self.last_timeout = timeout
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


def test_setting_a_temperature_waits_long_enough_for_a_slow_cloud_thermostat():
    session = FakeSession(FakeResponse())
    client = HAClient("http://ha", "token")
    client.session = session
    client.set_temperature("climate.lk_a", 21.0)
    assert (
        session.last_timeout >= 30
    )  # the LK integration answers only after its cloud call


def client_in_test_mode(session) -> HAClient:
    ha = HAClient("http://ha", "token", test_mode=True)
    ha.session = session
    return ha


def test_test_mode_sends_no_setpoint_to_home_assistant():
    session = FakeSession(FakeResponse())
    client_in_test_mode(session).set_temperature("climate.room_temp_setpoint", 16.0)
    assert session.calls == []


def test_test_mode_sends_no_hvac_mode_to_home_assistant():
    session = FakeSession(FakeResponse())
    client_in_test_mode(session).set_hvac_mode("climate.room_temp_setpoint", "off")
    assert session.calls == []


def test_test_mode_sends_no_notification_to_home_assistant():
    session = FakeSession(FakeResponse())
    HANotifier(client_in_test_mode(session)).notify("theria_x", "t", "m")
    assert session.calls == []


def test_test_mode_lets_the_nord_pool_read_through():
    session = FakeSession(FakeResponse({"service_response": {"SE4": []}}))
    answer = client_in_test_mode(session).get_service_response(
        "nordpool", "get_prices_for_date", {"date": "2026-11-05"}
    )
    assert answer == {"SE4": []}


def test_test_mode_refuses_a_service_with_an_answer_that_is_not_a_known_read():
    session = FakeSession(FakeResponse({"service_response": {}}))
    with pytest.raises(RuntimeError):
        client_in_test_mode(session).get_service_response("script", "turn_on", {})
    assert session.calls == []


def test_test_mode_reads_back_the_simulated_setpoint_on_the_real_state():
    session = FakeSession(
        FakeResponse({"state": "heat", "attributes": {"temperature": 20.0}})
    )
    ha = client_in_test_mode(session)
    ha.set_temperature("climate.room_temp_setpoint", 16.0)
    climate = ha.get_climate_state("climate.room_temp_setpoint")
    assert climate["target_temperature"] == 16.0 and climate["hvac_mode"] == "heat"


def test_a_simulated_setpoint_stays_on_its_own_entity():
    session = FakeSession(
        FakeResponse({"state": "heat", "attributes": {"temperature": 20.0}})
    )
    ha = client_in_test_mode(session)
    ha.set_temperature("climate.room_temp_setpoint", 16.0)
    assert ha.get_climate_state("climate.lk_a")["target_temperature"] == 20.0


def test_outside_test_mode_the_setpoint_reaches_home_assistant():
    session = FakeSession(FakeResponse())
    client(session).set_temperature("climate.room_temp_setpoint", 16.0)
    assert [call[0] for call in session.calls] == ["POST"]


def test_test_mode_is_off_unless_the_environment_turns_it_on():
    assert env_test_mode({}) is False
    assert env_test_mode({"HA_TEST_MODE": "false"}) is False
    assert env_test_mode({"HA_TEST_MODE": "true"}) is True
