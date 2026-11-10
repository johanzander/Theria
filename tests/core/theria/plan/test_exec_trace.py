from datetime import timedelta

from core.theria.plan.exec_trace import ExecTraceStore
from core.theria.plan.recent_log import RecentLines

from .test_day_view import PLAN_START
from .test_service_active import make, run_cycle
from .test_service_lk import LIMIT, LK

SINCE = PLAN_START - timedelta(hours=2)


def traced(settings, tmp_path, **kwargs):
    store = ExecTraceStore(str(tmp_path / "trace.jsonl"), PLAN_START)
    service, ha, *rest = make(settings, exec_trace=store, **kwargs)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    return service, ha, store


def test_every_control_cycle_keeps_what_was_read_asked_and_written(settings, tmp_path):
    _, ha, store = traced(settings, tmp_path)
    [record] = store.read(SINCE)
    assert record["mode"] == "active"
    assert record["pump"]["read"] == 18.0  # the setpoint the pump had
    assert record["pump"]["result"]["action"] in ("wrote", "kept")
    assert (
        record["lk"]["read"][LK] == 21.0
    )  # what each thermostat showed before the cycle
    assert record["lk"]["targets"][LK] == LIMIT  # what the cycle asked of it
    assert record["lk"]["result"]["action"] == "wrote"
    assert record["plan_hour"]["rts_c"] is not None and "closed" in record["plan_hour"]


def test_a_cycle_that_changes_nothing_is_kept_too(settings, tmp_path):
    service, _, store = traced(settings, tmp_path)
    run_cycle(service)
    records = store.read(SINCE)
    assert len(records) == 2
    assert records[-1]["lk"]["result"]["action"] == "kept"
    assert (
        records[-1]["lk"]["read"][LK] == LIMIT
    )  # the reading is there even when nothing was written


def test_a_trace_is_only_kept_in_active_mode(settings, tmp_path):
    store = ExecTraceStore(str(tmp_path / "trace.jsonl"), PLAN_START)
    service, *_ = make(settings, exec_trace=store)  # shadow
    run_cycle(service)
    assert store.read(SINCE) == []


def test_recent_lines_drop_what_is_older_than_the_window_and_skip_damaged_lines(
    tmp_path,
):
    path = tmp_path / "log.jsonl"
    store = RecentLines(str(path), PLAN_START, timedelta(days=1), "at")
    store.append({"at": (PLAN_START - timedelta(hours=2)).isoformat(), "n": 1})
    with open(path, "a") as f:
        f.write("{not json\n")
    again = RecentLines(
        str(path), PLAN_START + timedelta(hours=20), timedelta(days=1), "at"
    )
    assert [r["n"] for r in again.read(PLAN_START - timedelta(days=3))] == [1]
    later = RecentLines(
        str(path), PLAN_START + timedelta(days=2), timedelta(days=1), "at"
    )
    assert later.read(PLAN_START - timedelta(days=3)) == []
