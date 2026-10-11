from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import sensor_api
from tests.core.theria.test_sensor_guard import HEALTHY, history_for, make_guard


@pytest.fixture
def client():
    guard, ha = make_guard(HEALTHY, history_for())
    app = FastAPI()
    app.include_router(sensor_api.router)
    app.dependency_overrides[sensor_api.get_guard] = lambda: guard
    return TestClient(app), guard, ha


def test_status_is_empty_until_the_first_check(client):
    http, _, _ = client
    body = http.get("/api/comfort-sensors").json()
    assert body["status"] == "unknown" and body["sensors"] == []


def test_recheck_returns_every_sensor_with_status_age_battery_and_issues(client):
    http, _, ha = client
    ha.states["sensor.son_v_temp"] = ("unavailable", 5000)
    body = http.post("/api/comfort-sensors/recheck").json()
    assert body["status"] == "fault"
    by_name = {s["name"]: s for s in body["sensors"]}
    assert by_name["Vardagsrum LK"]["status"] == "ok"
    assert by_name["Vardagsrum LK"]["battery_percent"] == 80
    assert by_name["Vardagsrum Sonoff"]["issues"][0]["code"] == "unavailable"
    assert body["room_temperature_c"] == pytest.approx(20.0)
    assert datetime.fromisoformat(body["checked_at"]).tzinfo is not None
    assert http.get("/api/comfort-sensors").json()["status"] == "fault"


def test_no_usable_sensor_gives_a_null_room_temperature_and_a_reason(client):
    http, _, ha = client
    ha.states["sensor.lk_v_temp"] = ("unavailable", 5000)
    ha.states["sensor.son_v_temp"] = ("unavailable", 5000)
    body = http.post("/api/comfort-sensors/recheck").json()
    assert body["room_temperature_c"] is None
    assert "no healthy" in body["room_temperature_note"]


def test_unconfigured_guard_is_503():
    app = FastAPI()
    app.include_router(sensor_api.router)
    sensor_api.guard = None
    assert TestClient(app).get("/api/comfort-sensors").status_code == 503


def test_reading_the_status_never_triggers_a_new_check_of_home_assistant(client):
    http, guard, ha = client
    http.post("/api/comfort-sensors/recheck")
    reads_after_check = len(ha.reads)
    guard._now = lambda: datetime(
        2026, 10, 12, 13, 0, tzinfo=UTC
    )  # report now older than 5 min
    http.get("/api/comfort-sensors")
    http.get("/api/comfort-sensors")
    assert len(ha.reads) == reads_after_check
