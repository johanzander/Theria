from datetime import UTC, datetime, timedelta

import pytest

from core.theria import influxdb_helper
from core.theria.exceptions import ModelFitError
from core.theria.plan import calibration_data
from core.theria.plan.calibration_data import InfluxHistorySource

START = datetime(2026, 1, 1, tzinfo=UTC)
STOP = START + timedelta(days=1)


class FakeResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


@pytest.fixture
def influx_config(monkeypatch):
    monkeypatch.setattr(
        influxdb_helper,
        "get_influxdb_config",
        lambda: {"url": "http://influx", "username": "u", "password": "p"},
    )


def test_helper_marks_no_data_results_with_a_reason_code(monkeypatch, influx_config):
    for response in (FakeResponse(204), FakeResponse(200, "")):
        monkeypatch.setattr(
            influxdb_helper.requests, "post", lambda *a, r=response, **k: r
        )
        result = influxdb_helper.get_sensor_timeseries("outdoor", START, STOP)
        assert result["status"] == "error"
        assert result["reason"] == "no_data"


def test_helper_server_errors_have_no_no_data_reason(monkeypatch, influx_config):
    monkeypatch.setattr(
        influxdb_helper.requests, "post", lambda *a, **k: FakeResponse(500)
    )
    result = influxdb_helper.get_sensor_timeseries("outdoor", START, STOP)
    assert result["status"] == "error" and "reason" not in result


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
