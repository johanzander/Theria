"""The heat pump timeline: what the Värmepump tab draws, read from a fake InfluxDB."""

import re
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import heat_pump_api
from core.theria import influxdb_helper

START = datetime(2026, 11, 8, 0, 0, tzinfo=UTC)
HOUR = 3600


def epoch(moment: datetime) -> int:
    return int(moment.timestamp())


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class FakeInflux:
    """Answers the InfluxQL the code sends, from points per measurement."""

    def __init__(self):
        self.points = defaultdict(list)

    def add(self, measurement, moment, value):
        self.points[measurement].append((epoch(moment), value))

    def _statement(self, statement):
        name = re.search(r'FROM "([^"]+)"', statement)[1]
        rows = sorted(self.points[name])
        for pattern, keep in (
            (r"time >= (-?\d+)s", lambda t, v, x: t >= x),
            (r"time < (-?\d+)s", lambda t, v, x: t < x),
            (r'"\w+" >= (-?[\d.]+)', lambda t, v, x: v >= x),
            (r'"\w+" <= (-?[\d.]+)', lambda t, v, x: v <= x),
        ):
            found = re.search(pattern, statement)
            if found:
                limit = float(found[1])
                rows = [(t, v) for t, v in rows if keep(t, v, limit)]
        group = re.search(r"GROUP BY time\((\d+)s\)", statement)
        if group:
            step = int(group[1])
            aggregate = re.search(r"SELECT (\w+)\(", statement)[1]
            buckets = defaultdict(list)
            for t, v in rows:
                buckets[t - t % step].append(v)
            how = {
                "mean": lambda v: sum(v) / len(v),
                "last": lambda v: v[-1],
                "max": max,
            }
            rows = [(t, how[aggregate](buckets[t])) for t in sorted(buckets)]
        if "DESC" in statement:
            rows = rows[::-1]
        limit = re.search(r"LIMIT (\d+)", statement)
        if limit:
            rows = rows[: int(limit[1])]
        values = [[t * 1000, v] for t, v in rows]
        return {"series": [{"values": values}]} if values else {}

    def get(self, url, auth, params, timeout):
        statements = [s for s in params["q"].split(";") if s.strip()]
        return FakeResponse(200, {"results": [self._statement(s) for s in statements]})


@pytest.fixture
def influx(monkeypatch):
    fake = FakeInflux()
    monkeypatch.setattr(influxdb_helper.requests, "get", fake.get)
    monkeypatch.setattr(
        influxdb_helper,
        "get_influxdb_config",
        lambda: {
            "url": "http://db/query",
            "username": "u",
            "password": "p",
            "bucket": "ha/autogen",
        },
    )
    return fake


@pytest.fixture
def http():
    app = FastAPI()
    app.include_router(heat_pump_api.router)
    return TestClient(app)


def timeline(http, start, stop):
    response = http.get(
        "/api/heatpump/timeline",
        params={"start": start.isoformat(), "stop": stop.isoformat()},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_a_value_stored_only_on_change_is_drawn_from_the_start_of_the_range(
    http, influx
):
    influx.add("sensor.heating_setpoint", START - timedelta(days=20), 38.0)
    body = timeline(http, START, START + timedelta(hours=2))
    assert set(body["series"]["flowset"]) == {38.0}


def test_a_sensor_without_readings_has_no_values_until_its_first(http, influx):
    influx.add("sensor.outdoor", START + timedelta(hours=1), 7.0)
    body = timeline(http, START, START + timedelta(hours=2))
    assert body["series"]["outdoor"][0] is None
    assert body["series"]["outdoor"][-1] == 7.0


def test_a_garbage_reading_does_not_pull_the_value_it_shares_a_bucket_with(
    http, influx
):
    moment = START + timedelta(minutes=30)
    influx.add("sensor.outdoor", moment, 7.0)
    influx.add("sensor.outdoor", moment + timedelta(seconds=5), 240.0)
    body = timeline(http, START, START + timedelta(days=7))
    assert set(body["series"]["outdoor"]) - {None} == {7.0}


def test_a_garbage_reading_before_the_range_is_not_carried_into_it(http, influx):
    influx.add("sensor.heating_setpoint", START - timedelta(hours=3), 37.0)
    influx.add("sensor.heating_setpoint", START - timedelta(hours=1), 0.0)
    body = timeline(http, START, START + timedelta(hours=1))
    assert set(body["series"]["flowset"]) == {37.0}


def test_the_delta_is_forward_minus_return(http, influx):
    influx.add("sensor.heat_carrier_forwrd", START + timedelta(seconds=10), 43.0)
    influx.add("sensor.heat_carrier_return", START + timedelta(seconds=10), 36.0)
    body = timeline(http, START, START + timedelta(hours=1))
    assert body["series"]["hc_delta"][0] == pytest.approx(7.0)


def test_a_week_is_shown_coarser_than_a_day_and_stays_drawable(http, influx):
    day = timeline(http, START - timedelta(days=1), START)
    week = timeline(http, START - timedelta(days=7), START)
    assert week["step_ms"] > day["step_ms"]
    assert len(week["times"]) < 2500
    assert all(len(values) == len(week["times"]) for values in week["series"].values())


def test_energy_per_hour_is_the_rise_of_the_counter(http, influx):
    counter = "sensor.supp_energy_heating"
    influx.add(counter, START - timedelta(minutes=10), 100.0)
    influx.add(counter, START + timedelta(minutes=30), 105.0)
    influx.add(counter, START + timedelta(minutes=90), 112.0)
    body = timeline(http, START, START + timedelta(hours=2))
    assert body["energy"]["series"]["supp_heat"] == [5.0, 7.0]


def test_a_garbage_counter_reading_gives_neither_a_negative_hour_nor_a_lost_kwh(
    http, influx
):
    counter = "sensor.compr_cons_heating"
    influx.add(counter, START - timedelta(minutes=10), 100.0)
    influx.add(counter, START + timedelta(minutes=30), 101.0)
    influx.add(counter, START + timedelta(minutes=90), 0.0)
    influx.add(counter, START + timedelta(minutes=150), 104.0)
    values = timeline(http, START, START + timedelta(hours=3))["energy"]["series"][
        "cons_heat"
    ]
    assert min(values) >= 0
    assert sum(values) == pytest.approx(4.0)


@pytest.mark.parametrize("days", [30, 60])
def test_a_long_range_sums_the_energy_per_local_day(http, influx, days):
    start = START - timedelta(days=days)
    for day in range(days + 2):
        influx.add(
            "sensor.supp_energy_heating", start + timedelta(days=day), 1000.0 + 10 * day
        )
    energy = timeline(http, start, START)["energy"]
    local = [
        datetime.fromtimestamp(t / 1000, ZoneInfo("Europe/Stockholm"))
        for t in energy["times"]
    ]
    assert {moment.hour for moment in local} == {0}
    assert 8 <= energy["series"]["supp_heat"][10] <= 12


def test_the_price_is_read_per_quarter_and_carried_until_it_changes(http, influx):
    sensor = "sensor.nord_pool_se4_current_price"
    influx.add(sensor, START - timedelta(minutes=5), 1.5)
    influx.add(sensor, START + timedelta(minutes=30), 2.5)
    prices = timeline(http, START, START + timedelta(hours=1))["prices"]
    assert prices["values"] == [1.5, 1.5, 2.5, 2.5]
    assert prices["times"][1] - prices["times"][0] == 15 * 60 * 1000


def test_a_stop_in_the_future_ends_the_series_now(http, influx):
    now = datetime.now(UTC)
    body = timeline(http, now - timedelta(hours=1), now + timedelta(days=1))
    assert body["stop_ms"] <= epoch(now + timedelta(minutes=1)) * 1000
    assert body["times"][-1] <= body["stop_ms"]


@pytest.mark.parametrize(
    ("start", "stop"),
    [
        (START, START - timedelta(hours=1)),
        (START - timedelta(days=400), START),
    ],
)
def test_a_reversed_or_too_long_range_is_rejected(http, influx, start, stop):
    response = http.get(
        "/api/heatpump/timeline",
        params={"start": start.isoformat(), "stop": stop.isoformat()},
    )
    assert response.status_code == 422


class FakeHA:
    def __init__(self, states):
        self.states = states

    def get_state(self, entity_id):
        if entity_id not in self.states:
            raise ValueError(f"Entity not found: {entity_id}")
        return {"state": self.states[entity_id]}


def pump_now(states):
    app = FastAPI()
    app.include_router(heat_pump_api.router)
    app.dependency_overrides[heat_pump_api.get_ha_client] = lambda: FakeHA(states)
    response = TestClient(app).get("/api/heatpump/now")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_pump_now_is_what_home_assistant_reads(influx):
    body = pump_now({"sensor.outdoor": "7.10", "sensor.compressor_speed_2": "30.00"})
    assert body["values"]["outdoor"] == 7.1
    assert body["values"]["comp"] == 30.0


def test_an_unavailable_sensor_has_no_value_now(influx):
    assert pump_now({"sensor.outdoor": "unavailable"})["values"]["outdoor"] is None


def test_a_garbage_reading_is_not_shown_as_the_value_now(influx):
    assert pump_now({"sensor.outdoor": "240.0"})["values"]["outdoor"] is None


@pytest.mark.parametrize(("valve", "hot_water"), [("on", True), ("off", False)])
def test_the_valve_tells_whether_the_pump_makes_hot_water(influx, valve, hot_water):
    assert pump_now({"binary_sensor.switch_valve_1": valve})["hot_water"] is hot_water


def test_the_pump_now_needs_home_assistant(http, influx):
    assert http.get("/api/heatpump/now").status_code == 503


def test_a_database_that_cannot_answer_is_a_bad_gateway(http, influx, monkeypatch):
    def fail(url, auth, params, timeout):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(influxdb_helper.requests, "get", fail)
    response = http.get(
        "/api/heatpump/timeline",
        params={
            "start": START.isoformat(),
            "stop": (START + timedelta(hours=1)).isoformat(),
        },
    )
    assert response.status_code == 502
