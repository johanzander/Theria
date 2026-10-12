import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api

from .test_service import fitted_models
from .test_service_active import make

LK = "climate.lk_bibliotek_thermostat"


def ready(settings, **kwargs):
    service, ha, *_ = make(settings, **kwargs)
    service._apply_models(fitted_models())
    service.run_once()
    return service, ha


def statuses(service):
    return {c.name: c.status for c in service.preflight()}


def test_a_healthy_system_has_no_failing_check(settings):
    service, _ = ready(settings)
    assert "fail" not in statuses(service).values()


def test_models_that_are_not_fitted_fail_the_check(settings):
    service, *_ = make(settings)
    service.run_once()
    assert statuses(service)["models"] == "fail"


def test_a_missing_or_old_plan_fails_the_check(settings):
    service, _ = ready(settings)
    service._snapshots.clear()
    assert statuses(service)["plan"] == "fail"


def test_the_old_automation_being_on_fails_the_check(settings):
    service, ha = ready(settings)
    ha.automation = "on"
    assert statuses(service)["automation"] == "fail"


def test_a_pump_setpoint_outside_the_plans_range_fails_the_check(settings):
    service, _ = ready(settings, setpoint=22.0)
    assert statuses(service)["pump setpoint"] == "fail"


def test_an_unreadable_thermostat_fails_the_check(settings):
    service, ha = ready(settings)
    ha.get_climate_state = lambda entity: (
        (_ for _ in ()).throw(RuntimeError("down"))
        if entity == LK
        else {"target_temperature": 18.0}
    )
    assert statuses(service)["LK thermostats"] == "fail"


def test_a_plan_that_breaks_the_floor_only_warns(settings):
    service, _ = ready(settings)
    snapshot = service.latest()
    object.__setattr__(snapshot.plan, "floor_violation_hours", 3)
    assert statuses(service)["plan comfort"] == "warn"


@pytest.fixture
def client(settings):
    service, ha = ready(settings)
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    return TestClient(app), service, ha


def test_the_preflight_endpoint_lists_the_checks(client):
    http, _, _ = client
    body = http.get("/api/plan/preflight").json()
    assert body["ok"] is True
    assert {c["name"] for c in body["checks"]} >= {"models", "plan", "automation"}


def test_active_mode_is_refused_over_the_api_while_a_check_fails(client):
    http, service, _ = client
    ok = http.post("/api/plan/mode", json={"mode": "active", "confirm": True})
    assert ok.status_code == 200
    http.post("/api/plan/mode", json={"mode": "off"})
    service._snapshots.clear()  # a failure
    refused = http.post("/api/plan/mode", json={"mode": "active", "confirm": True})
    assert refused.status_code == 422 and "plan" in refused.json()["detail"]
    assert service.mode == "off"
