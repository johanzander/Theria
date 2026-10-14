from dataclasses import replace
from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api
from core.theria.plan.executor import PlanExecutor
from core.theria.plan.service import PlanService

from .test_service import NOW, FakeWeather, fitted_models, price_attributes
from .test_service_active import FakeNotifier, FakeSafety, WritableHA, good_days


@pytest.fixture
def client(settings):
    ha = WritableHA(price_attributes(), 17.0)
    service = PlanService(
        ha,
        replace(settings, mode="shadow"),
        FakeWeather(),
        now=lambda: NOW,
        executor=PlanExecutor(ha, settings, now=lambda: NOW),
        safety=FakeSafety(),
        notifier=FakeNotifier(),
        outcome_loader=lambda start, stop: good_days(date(2026, 10, 6)),
    )
    service._apply_models(
        fitted_models()
    )  # a healthy system: models fitted, plan fresh
    service.run_once()
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    return TestClient(app), service, ha


def test_active_mode_needs_confirm_in_the_request(client):
    http, service, _ = client
    assert http.post("/api/plan/mode", json={"mode": "active"}).status_code == 422
    assert service.mode == "shadow"
    ok = http.post("/api/plan/mode", json={"mode": "active", "confirm": True})
    assert ok.json() == {"mode": "active"}


def test_leaving_active_through_the_api_restores_the_base(client):
    http, _, ha = client
    http.post("/api/plan/mode", json={"mode": "active", "confirm": True})
    http.post("/api/plan/mode", json={"mode": "off"})
    assert ha.setpoint == 18.0


def test_execution_reports_the_last_write(client):
    http, service, _ = client
    http.post("/api/plan/mode", json={"mode": "active", "confirm": True})
    service.run_once()
    service._control()
    body = http.get("/api/plan/execution").json()
    assert body["mode"] == "active"
    assert "stage" not in body
    assert body["authority"] == 0.5 and body["radiators"] is None
    assert body["regulator"] == []
    assert body["last_execution"]["action"] in ("wrote", "kept")


def test_savings_lists_days_and_a_total(client):
    http, _, _ = client
    body = http.get("/api/savings?days=5").json()
    assert len(body["days"]) == 5
    assert body["total_saving_sek"] == pytest.approx(
        sum(d["saving_sek"] for d in body["days"])
    )
    assert body["from"] and body["to"]


def test_savings_is_unavailable_without_an_outcome_source(settings):
    from .test_service import make_service

    service = make_service(settings)
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    assert TestClient(app).get("/api/savings").status_code == 409


def test_a_failed_restore_is_reported_as_a_bad_gateway_and_the_mode_stays(client):
    http, service, ha = client
    http.post("/api/plan/mode", json={"mode": "active", "confirm": True})
    ha.fail_writes = True
    response = http.post("/api/plan/mode", json={"mode": "off"})
    assert response.status_code == 502
    assert service.mode == "active"
