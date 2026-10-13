import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api
from core.theria.exceptions import ModelFitError
from tests.core.theria.plan.test_service import (
    FakeCalibrator,
    fitted_models,
    make_service,
)


@pytest.fixture
def client(settings):
    service = make_service(settings, calibrator=FakeCalibrator(fitted_models()))
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    return TestClient(app), service


def test_plan_is_404_until_one_is_computed(client):
    http, service = client
    assert http.get("/api/plan").status_code == 404
    service.run_once()
    body = http.get("/api/plan").json()
    assert len(body["plan"]["hours"]) == 36
    assert body["plan"]["hours"][0]["reason"]["binding"]
    assert body["mode"] == "shadow"
    assert body["age_seconds"] == 0


def test_history_lists_summaries(client):
    http, service = client
    service.run_once()
    body = http.get("/api/plan/history?hours=24").json()
    assert len(body) == 1
    assert set(body[0]) >= {
        "computed_at",
        "rts_now_c",
        "objective_sek",
        "baseline_objective_sek",
    }


def test_history_rejects_out_of_range_hours(client):
    http, _ = client
    assert http.get("/api/plan/history?hours=0").status_code == 422


def test_model_health_reports_priors(client):
    http, _ = client
    body = http.get("/api/model/health").json()
    assert body["models"]["pump"]["source"] == "prior"
    assert body["models"]["room"]["source"] == "prior"


def test_mode_can_be_switched_but_active_is_refused(client):
    http, service = client
    assert http.post("/api/plan/mode", json={"mode": "off"}).json() == {"mode": "off"}
    assert service.mode == "off"
    assert http.post("/api/plan/mode", json={"mode": "active"}).status_code == 422
    assert service.mode == "off"


def test_unconfigured_service_is_503():
    app = FastAPI()
    app.include_router(plan_api.router)
    plan_api.service = None
    assert TestClient(app).get("/api/plan").status_code == 503


def test_model_health_includes_fit_metadata_and_scores(client):
    http, service = client
    service.run_once()
    service.refit()
    body = http.get("/api/model/health").json()
    assert body["models"]["pump"]["source"] == "fitted"
    assert body["models"]["pump"]["fit"]["flow_rows"] == 400
    assert body["models"]["last_fit_error"] is None
    assert isinstance(body["scores"], list)


def test_refit_endpoint_returns_the_fitted_models(client):
    http, _ = client
    response = http.post("/api/model/refit")
    assert response.status_code == 200
    assert response.json()["room"]["response"]["gain_c_per_c"] == 0.39


def test_rejected_refit_is_422_and_says_the_old_models_are_kept(settings):
    service = make_service(
        settings, calibrator=FakeCalibrator(error=ModelFitError("too little data"))
    )
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    response = TestClient(app).post("/api/model/refit")
    assert response.status_code == 422
    assert "previous models kept" in response.json()["detail"]


def test_refit_without_calibration_is_409(settings):
    service = make_service(settings)
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    assert TestClient(app).post("/api/model/refit").status_code == 409


def test_day_is_404_until_a_plan_exists_then_lists_the_whole_day(client):
    http, service = client
    assert http.get("/api/plan/day").status_code == 404
    service.run_once()
    body = http.get("/api/plan/day").json()
    assert len(body["hours"]) == 24
    assert any(not h["past"] for h in body["hours"])


def test_day_rejects_a_malformed_date(client):
    http, service = client
    service.run_once()
    assert http.get("/api/plan/day?day=tomorrow").status_code == 422


def test_tuning_lists_the_settings_with_advice_and_applies_nothing(client):
    http, service = client
    service.run_once()
    body = http.get("/api/plan/tuning").json()
    keys = [r["key"] for r in body]
    assert keys[:2] == ["floor_c", "max_flow_c"] and "stage" not in keys
    assert all(r["status"] in {"ok", "consider", "no data"} for r in body)
