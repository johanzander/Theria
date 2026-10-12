from dataclasses import replace
from datetime import UTC, date, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api
from core.theria.plan.night_drift import NightDrift
from core.theria.plan.parameter_log import ParameterLog

from .test_service import FakeCalibrator, fitted_models
from .test_service_active import good_days, make
from .test_service_morning import IN_THE_NIGHT, cold_mornings

NOW = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)
NIGHT = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 120, 0.1, ())


def test_records_are_appended_and_read_back_oldest_first(tmp_path):
    log = ParameterLog(str(tmp_path / "p.jsonl"))
    log.append(NOW, "refit", "nightly", {"gain": 0.45})
    log.append(NOW.replace(hour=4), "margin", "raise", {"gain": 0.45, "margin": 0.25})
    records = log.read()
    assert [r["kind"] for r in records] == ["refit", "margin"]
    assert (
        records[1]["at"] == NOW.replace(hour=4)
        and records[1]["params"]["margin"] == 0.25
    )


def test_the_history_survives_a_new_log_object_like_a_restart(tmp_path):
    path = str(tmp_path / "p.jsonl")
    ParameterLog(path).append(NOW, "refit", "", {"gain": 0.4})
    assert ParameterLog(path).read()[0]["params"]["gain"] == 0.4


def test_a_missing_file_is_an_empty_history_and_a_damaged_line_is_skipped(tmp_path):
    path = tmp_path / "p.jsonl"
    log = ParameterLog(str(path))
    assert log.read() == []
    log.append(NOW, "refit", "", {"gain": 0.4})
    with open(path, "a") as f:
        f.write("not json at all\n")
    log.append(NOW, "refit", "", {"gain": 0.41})
    assert [r["params"]["gain"] for r in log.read()] == [0.4, 0.41]


def test_read_returns_only_the_latest_records_when_limited(tmp_path):
    log = ParameterLog(str(tmp_path / "p.jsonl"))
    for i in range(10):
        log.append(NOW, "refit", str(i), {"i": i})
    assert [r["note"] for r in log.read(limit=3)] == ["7", "8", "9"]


def service_with_log(settings, tmp_path, **kwargs):
    log = ParameterLog(str(tmp_path / "p.jsonl"))
    service, *_ = make(settings, **kwargs)
    service._parameter_log = log
    return service, log


def test_every_refit_records_all_the_learned_parameters(settings, tmp_path):
    service, log = service_with_log(settings, tmp_path)
    service._calibrator = FakeCalibrator(replace(fitted_models(), night_drift=NIGHT))
    service.refit()
    record = log.read()[-1]
    assert record["kind"] == "refit"
    params = record["params"]
    for key in (
        "flow_intercept",
        "flow_slope",
        "energy_per_rts",
        "aux_per_flow_c",
        "room_gain_c_per_c",
        "room_time_constant_days",
        "night_intercept_c",
        "night_slope_c_per_c",
        "night_nights",
        "morning_margin_c",
        "stage",
    ):
        assert key in params, key
    assert params["night_slope_c_per_c"] == 0.06


def test_a_change_of_the_night_margin_is_recorded_with_the_reason(settings, tmp_path):
    from dataclasses import replace as r

    night = r(settings, night_start_hour=0, night_end_hour=6)
    service, log = service_with_log(
        night, tmp_path, clock=lambda: IN_THE_NIGHT, outcomes=cold_mornings()
    )
    service.set_mode("active", confirm=True)
    service._control()
    records = [x for x in log.read() if x["kind"] == "margin"]
    assert len(records) == 1 and records[0]["params"][
        "morning_margin_c"
    ] == pytest.approx(0.25)
    assert "night margin" in records[0]["note"]


def test_a_stage_change_is_recorded(settings, tmp_path):
    from core.theria.plan.stages import StageState
    from core.theria.plan.state_store import PlanState, PlanStateStore

    store = PlanStateStore(str(tmp_path / "state.json"))
    store.save(
        PlanState(mode="active", models=None, stage=StageState(0, date(2026, 10, 6)))
    )
    service, log = service_with_log(
        settings, tmp_path, outcomes=good_days(date(2026, 10, 6)), state_store=store
    )
    service._control()
    assert any(x["kind"] == "stage" and x["params"]["stage"] == 2 for x in log.read())


def test_nothing_breaks_without_a_log(settings):
    service, *_ = make(settings)
    service._calibrator = FakeCalibrator(fitted_models())
    service.refit()  # no parameter log configured


def test_the_history_is_available_over_the_api(settings, tmp_path):
    service, log = service_with_log(settings, tmp_path)
    log.append(NOW, "refit", "nightly", {"gain": 0.45})
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    body = TestClient(app).get("/api/plan/parameters?limit=5").json()
    assert (
        body["history"][0]["kind"] == "refit"
        and body["history"][0]["params"]["gain"] == 0.45
    )
    assert "flow_slope" in body["current"]
