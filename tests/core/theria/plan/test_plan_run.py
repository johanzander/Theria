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
            "shift_min_gain_sek_per_kwh=3",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "replay matches the stored decision: True" in out
    assert "with shift_min_gain_sek_per_kwh=3" in out


def test_a_run_keeps_the_room_and_flow_it_expected_so_they_can_be_checked_later(
    settings, tmp_path
):
    service, store = kept_run(settings, tmp_path)
    [body] = store.read(PLAN_START - timedelta(hours=1))
    hour = service.latest().plan.hours[0]
    assert body["decided"][0]["room_c"] == hour.room_c_end + hour.sun_swing_c
    assert body["decided"][0]["flow_c"] == hour.flow_target_c


def test_a_run_keeps_the_energy_table(settings, tmp_path):
    from dataclasses import replace

    import numpy as np

    from core.theria.plan.plan_run import run_to_dict

    _, store = kept_run(settings, tmp_path)
    [body] = store.read(PLAN_START - timedelta(hours=1))
    run = run_from_dict(body)
    n = len(run.inputs.hour_starts)
    inputs = replace(
        run.inputs,
        energy_setpoints_c=np.array([17.0, 18.0]),
        energy_kwh_table=np.ones((n, 2)),
    )
    again = run_from_dict(run_to_dict(replace(run, inputs=inputs)))
    assert again.inputs.energy_kwh_table.shape == (n, 2)
    assert list(again.inputs.energy_setpoints_c) == [17.0, 18.0]
