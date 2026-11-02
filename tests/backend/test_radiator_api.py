import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import radiator_api
from core.theria.plan.event_log import EventLog
from tests.core.theria.radiators.test_service import make, prices


@pytest.fixture
def client(tmp_path):
    service, _ = make(
        periods=prices(dear_at={3}), log=EventLog(str(tmp_path / "events.jsonl"))
    )
    app = FastAPI()
    app.include_router(radiator_api.router)
    app.dependency_overrides[radiator_api.get_service] = lambda: service
    return TestClient(app), service


def test_status_is_404_until_a_cycle_has_run(client):
    http, service = client
    assert http.get("/api/radiators/status").status_code == 404
    service.run_once()
    assert http.get("/api/radiators/status").status_code == 200


def test_status_returns_would_write_and_availability(client):
    http, service = client
    service.run_once()
    body = http.get("/api/radiators/status").json()
    assert set(body["would_write"]) == {
        "climate.vantsidan",
        "climate.klippsidan",
        "climate.hallway",
        "climate.kitchen",
    }
    assert body["availability"]["climate.hallway"] is True
    assert body["age_seconds"] == 0
    assert body["availability_share"]["climate.hallway"] == 1.0
    assert body["salon"]["reading"]["mean_c"] == pytest.approx(19.825)


def test_prefs_roundtrip_through_put_and_get(client):
    http, _ = client
    first = http.get("/api/radiators/prefs").json()
    assert first["is_default"] is True
    put = http.put(
        "/api/radiators/prefs", json={"values": {"salon_open_target_c": 19.5}}
    )
    assert put.status_code == 200
    again = http.get("/api/radiators/prefs").json()
    assert again["prefs"]["salon_open_target_c"] == 19.5
    assert again["defaults"]["salon_open_target_c"] == 19.0
    assert again["is_default"] is False


def test_invalid_prefs_are_422_with_a_message(client):
    http, _ = client
    r = http.put("/api/radiators/prefs", json={"values": {"salon_preheat_c": 10.0}})
    assert r.status_code == 422
    assert "preheat" in r.json()["detail"]
    assert http.get("/api/radiators/prefs").json()["is_default"] is True


def test_delete_resets_to_defaults(client):
    http, _ = client
    http.put("/api/radiators/prefs", json={"values": {"salon_open_target_c": 19.5}})
    assert http.delete("/api/radiators/prefs").status_code == 200
    assert http.get("/api/radiators/prefs").json()["is_default"] is True


def test_log_lists_the_shadow_lines(client):
    http, service = client
    service.run_once()
    [line] = http.get("/api/radiators/log?limit=5").json()
    assert line["kind"] == "radiators"


def test_503_when_not_configured():
    app = FastAPI()
    app.include_router(radiator_api.router)
    radiator_api.service = None
    assert TestClient(app).get("/api/radiators/status").status_code == 503


def test_status_carries_the_last_error_after_a_failed_cycle(client):
    http, service = client
    service.run_once()
    service._ha.states["sensor.outdoor"] = {"state": "unavailable", "attributes": {}}
    service._outdoor_c = None  # as if the service had just started
    import asyncio

    asyncio.run(service.cycle())
    body = http.get("/api/radiators/status").json()
    assert "outdoor" in body["last_error"]


def test_a_non_numeric_room_temperature_is_422_not_500(client):
    http, _ = client
    r = http.put(
        "/api/radiators/prefs",
        json={"values": {"guest_rooms": [{"entity": "climate.a", "t_room_c": "abc"}]}},
    )
    assert r.status_code == 422
