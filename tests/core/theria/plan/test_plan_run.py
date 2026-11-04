import json
from datetime import timedelta
from subprocess import run as run_script
from sys import executable

from core.theria.plan.plan_run import PlanRunStore, replay, run_from_dict

from .test_day_view import PLAN_START
from .test_service_active import make, run_cycle


def kept_run(settings, tmp_path):
    store = PlanRunStore(str(tmp_path / "runs.jsonl"), PLAN_START)
    service, *_ = make(settings, plan_runs=store)
    run_cycle(service)
    return service, store


def test_a_kept_run_replays_to_the_plan_the_service_computed(settings, tmp_path):
    service, store = kept_run(settings, tmp_path)
    [body] = store.read(PLAN_START - timedelta(hours=1))
    again = replay(run_from_dict(body))
    computed = service.latest().plan
    assert [h.rts_c for h in again.hours] == [h.rts_c for h in computed.hours]
    assert again.objective_sek == computed.objective_sek


def test_a_run_survives_the_file(settings, tmp_path):
    _, store = kept_run(settings, tmp_path)
    reread = PlanRunStore(str(tmp_path / "runs.jsonl"), PLAN_START).read(
        PLAN_START - timedelta(hours=1)
    )
    assert reread == store.read(PLAN_START - timedelta(hours=1))
    assert replay(run_from_dict(reread[0])).hours


def test_runs_older_than_a_week_are_dropped_when_the_store_opens(settings, tmp_path):
    kept_run(settings, tmp_path)
    store = PlanRunStore(str(tmp_path / "runs.jsonl"), PLAN_START + timedelta(days=8))
    assert store.read(PLAN_START - timedelta(days=1)) == []
    assert (tmp_path / "runs.jsonl").read_text() == ""


def test_a_damaged_line_is_skipped(settings, tmp_path):
    kept_run(settings, tmp_path)
    with open(tmp_path / "runs.jsonl", "a") as f:
        f.write("{not json\n")
    store = PlanRunStore(str(tmp_path / "runs.jsonl"), PLAN_START)
    assert len(store.read(PLAN_START - timedelta(hours=1))) == 1


def test_the_replay_script_matches_the_stored_decision_and_takes_a_setting(
    settings, tmp_path
):
    _, store = kept_run(settings, tmp_path)
    [body] = store.read(PLAN_START - timedelta(hours=1))
    path = tmp_path / "runs.json"
    path.write_text(json.dumps([body]))
    out = run_script(
        [
            executable,
            "scripts/replay_plan.py",
            str(path),
            "--at",
            body["computed_at"],
            "--set",
            "comfort_cost_sek_per_c_hour=3",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "replay matches the stored decision: True" in out
    assert "with comfort_cost_sek_per_c_hour=3" in out


def test_a_run_keeps_the_options_of_every_decided_hour_and_the_valve_state_it_started_from(
    settings, tmp_path
):
    service, store = kept_run(settings, tmp_path)
    [body] = store.read(PLAN_START - timedelta(hours=1))
    first = body["decided"][0]
    assert (
        first["alternatives"] == service.latest().plan.hours[0].reason["alternatives"]
    )
    assert "held_by_commitment" in first
    assert body["inputs"]["previous_closed"] is False
    assert run_from_dict(body).inputs.previous_closed is False
