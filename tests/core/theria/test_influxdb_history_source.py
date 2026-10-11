from datetime import UTC, datetime, timedelta

import pytest

from core.theria import influxdb_helper
from core.theria.exceptions import ModelFitError
from core.theria.plan import calibration_data
from core.theria.plan.calibration_data import InfluxHistorySource

START = datetime(2026, 1, 1, tzinfo=UTC)
STOP = START + timedelta(days=1)


class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def series(name, columns, values):
    return {
        "results": [
            {
                "statement_id": 0,
                "series": [{"name": name, "columns": columns, "values": values}],
            }
        ]
    }


@pytest.fixture
def influx_config(monkeypatch):
    monkeypatch.setattr(
        influxdb_helper,
        "get_influxdb_config",
        lambda: {"url": "http://influx/query", "username": "u", "password": "p"},
    )


def serve(monkeypatch, response):
    calls = []

    def fake_get(url, **kwargs):
        calls.append((url, kwargs))
        return response

    monkeypatch.setattr(influxdb_helper.requests, "get", fake_get)
    return calls


def test_helper_asks_influxql_for_the_entity_between_the_two_times(
    monkeypatch, influx_config
):
    calls = serve(
        monkeypatch,
        FakeResponse(
            200, series("sensor.outdoor", ["time", "value"], [[1767225600000, 4.5]])
        ),
    )
    result = influxdb_helper.get_sensor_timeseries("outdoor", START, STOP)
    url, kwargs = calls[0]
    assert url == "http://influx/query" and kwargs["auth"] == ("u", "p")
    assert (
        kwargs["params"]["db"] == "home_assistant" and kwargs["params"]["epoch"] == "ms"
    )
    query = kwargs["params"]["q"]
    assert 'SELECT "value" FROM "sensor.outdoor"' in query
    assert (
        "time >= '2026-01-01T00:00:00Z'" in query
        and "time < '2026-01-02T00:00:00Z'" in query
    )
    assert result["status"] == "success"
    moment, value = result["data"][0]
    assert moment == START and value == 4.5


def test_helper_reads_attribute_fields_and_string_values(monkeypatch, influx_config):
    calls = serve(
        monkeypatch,
        FakeResponse(
            200,
            series(
                "climate.x", ["time", "hvac_action_str"], [[1767225600000, "heating"]]
            ),
        ),
    )
    result = influxdb_helper.get_sensor_timeseries(
        "x",
        START,
        STOP,
        domain="climate",
        field_name="hvac_action_str",
        parse_as_string=True,
    )
    assert 'SELECT "hvac_action_str" FROM "climate.x"' in calls[0][1]["params"]["q"]
    assert result["data"][0][1] == "heating"


def test_helper_marks_no_data_results_with_a_reason_code(monkeypatch, influx_config):
    serve(monkeypatch, FakeResponse(200, {"results": [{"statement_id": 0}]}))
    result = influxdb_helper.get_sensor_timeseries("outdoor", START, STOP)
    assert result["status"] == "error"
    assert result["reason"] == "no_data"


def test_helper_server_errors_have_no_no_data_reason(monkeypatch, influx_config):
    serve(monkeypatch, FakeResponse(500))
    result = influxdb_helper.get_sensor_timeseries("outdoor", START, STOP)
    assert result["status"] == "error" and "reason" not in result


def test_a_query_error_reported_in_the_body_is_not_no_data(monkeypatch, influx_config):
    serve(
        monkeypatch,
        FakeResponse(
            200, {"results": [{"statement_id": 0, "error": "database not found"}]}
        ),
    )
    result = influxdb_helper.get_sensor_timeseries("outdoor", START, STOP)
    assert result["status"] == "error" and "reason" not in result


def test_names_that_could_change_the_query_are_refused_before_any_request(
    monkeypatch, influx_config
):
    calls = serve(monkeypatch, FakeResponse(200, {"results": [{}]}))
    result = influxdb_helper.get_sensor_timeseries('x" OR 1=1 --', START, STOP)
    assert result["status"] == "error" and calls == []


def test_history_source_turns_no_data_into_an_empty_series(monkeypatch):
    monkeypatch.setattr(
        calibration_data,
        "get_sensor_timeseries",
        lambda *a, **k: {"status": "error", "reason": "no_data", "message": "whatever"},
    )
    assert InfluxHistorySource().series("sensor.outdoor", START, STOP) == []


def test_history_source_raises_for_real_failures_whatever_the_message_says(monkeypatch):
    monkeypatch.setattr(
        calibration_data,
        "get_sensor_timeseries",
        lambda *a, **k: {"status": "error", "message": "No valid data points found"},
    )
    with pytest.raises(ModelFitError):
        InfluxHistorySource().series("sensor.outdoor", START, STOP)


def test_history_source_returns_points_as_floats(monkeypatch):
    monkeypatch.setattr(
        calibration_data,
        "get_sensor_timeseries",
        lambda *a, **k: {"status": "success", "data": [(START, 1)]},
    )
    assert InfluxHistorySource().series("sensor.outdoor", START, STOP) == [(START, 1.0)]
