from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api


class FakeForecaster:
    def view(self, days):
        return {"days": [], "summary": [], "asked": days}


def client():
    app = FastAPI()
    app.include_router(plan_api.router)
    return TestClient(app)


def test_the_energy_forecast_is_served():
    plan_api.energy_forecaster = FakeForecaster()
    assert client().get("/api/energy-forecast?days=30").json()["asked"] == 30
    plan_api.energy_forecaster = None


def test_it_is_503_until_configured():
    plan_api.energy_forecaster = None
    assert client().get("/api/energy-forecast").status_code == 503
