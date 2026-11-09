from datetime import UTC, datetime

from core.theria import influxdb_helper


class Response:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"results": [{}]}


def test_a_plausible_range_is_applied_before_the_hour_is_averaged(monkeypatch):
    asked = {}

    def get(url, auth, params, timeout):
        asked.update(params)
        return Response()

    monkeypatch.setattr(influxdb_helper.requests, "get", get)
    monkeypatch.setattr(
        influxdb_helper,
        "get_influxdb_config",
        lambda: {"url": "http://x", "username": "u", "password": "p", "bucket": "db"},
    )
    influxdb_helper.get_sensor_timeseries(
        "outdoor",
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        hourly_mean=True,
        plausible=(-35.0, 45.0),
    )
    assert '"value" >= -35.0' in asked["q"]
    assert '"value" <= 45.0' in asked["q"]
