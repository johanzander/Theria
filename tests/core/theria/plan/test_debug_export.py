from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend import plan_api
from core.theria.plan.debug_export import (
    key_findings,
    log_anomalies,
    redact,
    thermostat_mismatches,
    first_hour_flips,
)
from core.theria.plan.exec_trace import ExecTraceStore
from core.theria.plan.plan_run import PlanRunStore

from .test_day_view import PLAN_START
from .test_service_active import make, run_cycle

T0 = datetime(2026, 10, 31, 16, tzinfo=UTC)


def run_line(minutes, rts, closed=False):
    return {
        "computed_at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "decided": [{"start": T0.isoformat(), "rts_c": rts, "valves_closed": closed}],
    }


def trace_line(minutes, read, targets):
    return {
        "at": (T0 + timedelta(minutes=minutes)).isoformat(),
        "lk": {"read": read, "targets": targets, "result": {"action": "wrote"}},
    }


def test_a_first_hour_that_changes_between_replans_is_counted_as_a_flip():
    runs = [
        run_line(0, 16, True),
        run_line(15, 18),
        run_line(30, 16, True),
        run_line(45, 16, True),
    ]
    flips = first_hour_flips(runs)
    assert flips == [
        (T0 + timedelta(minutes=15), "16c", "18"),
        (T0 + timedelta(minutes=30), "18", "16c"),
    ]


def test_a_steady_plan_has_no_flips():
    assert first_hour_flips([run_line(0, 16), run_line(15, 16), run_line(30, 16)]) == []


def test_a_thermostat_that_shows_something_else_than_was_asked_a_cycle_ago_is_reported():
    trace = [
        trace_line(0, {"a": 22.0}, {"a": 17.85}),
        trace_line(15, {"a": 18.5}, {"a": 17.85}),  # it shows 18.5, 17.85 was asked
        trace_line(30, {"a": 17.8}, {"a": 17.85}),  # within a tenth: fine
    ]
    found = thermostat_mismatches(trace)
    assert len(found) == 1
    assert (
        found[0]["entity"] == "a"
        and found[0]["read"] == 18.5
        and found[0]["asked"] == 17.85
    )


def test_the_same_warning_over_and_over_is_one_line_with_a_count():
    lines = [
        "2026-10-31 22:11:00 | WARNING | core.theria.plan.lk_executor:192 - LK write failed: a",
        "2026-10-31 22:30:00 | WARNING | core.theria.plan.lk_executor:192 - LK write failed: a",
        "2026-10-31 22:31:00 | ERROR | core.theria.plan.service:99 - plan failed",
        "2026-10-31 22:32:00 | INFO | core.theria.plan.service:50 - fine",
    ]
    found = log_anomalies(lines)
    assert [(f["count"], f["level"]) for f in found] == [(2, "WARNING"), (1, "ERROR")]
    assert found[0]["first"].startswith("2026-10-31 22:11") and found[0][
        "last"
    ].startswith("2026-10-31 22:30")


def test_secrets_are_redacted_wherever_they_sit():
    body = {
        "influxdb": {"username": "u", "password": "hunter2"},
        "list": [{"api_token": "abc", "n": 1}],
    }
    cleaned = redact(body)
    assert (
        cleaned["influxdb"]["password"] == "***"
        and cleaned["influxdb"]["username"] == "u"
    )
    assert cleaned["list"][0]["api_token"] == "***" and cleaned["list"][0]["n"] == 1


def test_the_findings_name_each_problem_with_when_it_happened():
    findings = key_findings(
        runs=[run_line(0, 16, True), run_line(15, 18)],
        trace=[
            trace_line(0, {"a": 22.0}, {"a": 17.85}),
            trace_line(15, {"a": 18.5}, {"a": 17.85}),
        ],
        events=[
            {
                "at": T0.isoformat(),
                "kind": "lk",
                "action": "paused",
                "detail": "thermostat changed by hand",
            }
        ],
        log_lines=[],
        preflight=[("plan", "fail", "no plan has been computed")],
    )
    text = "\n".join(findings)
    assert (
        "flipped" in text
        and "18.5" in text
        and "paused" in text
        and "no plan has been computed" in text
    )


# ---- the report and the endpoint


def active_service(settings, tmp_path):
    runs = PlanRunStore(str(tmp_path / "runs.jsonl"), PLAN_START)
    trace = ExecTraceStore(str(tmp_path / "trace.jsonl"), PLAN_START)
    service, *_ = make(settings, plan_runs=runs, exec_trace=trace)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    return service


def test_the_report_has_the_sections_that_make_a_problem_reproducible(
    settings, tmp_path
):
    report = active_service(settings, tmp_path).debug_report()
    for heading in (
        "# Theria debug export",
        "## Key findings",
        "## System",
        "## Preflight",
        "## Settings",
        "## Entities",
        "## Plan runs",
        "## Execution trace",
        "## Action log",
        "## System logs",
    ):
        assert heading in report, heading
    assert "scripts/replay_plan.py" in report  # how to use the runs


def test_the_endpoint_serves_the_report_as_a_download(settings, tmp_path):
    service = active_service(settings, tmp_path)
    app = FastAPI()
    app.include_router(plan_api.router)
    app.dependency_overrides[plan_api.get_service] = lambda: service
    response = TestClient(app).get("/api/export-debug-data")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert response.text.startswith("# Theria debug export")
