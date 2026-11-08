from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.forecast_check import (
    ForecastChecker,
    ForecastCheckStore,
    Measured,
    check_hour,
    load_measured,
    summarize,
)

HOUR = datetime(2026, 11, 8, 10, tzinfo=UTC)


def run(computed_at, first, rows):
    """A stored plan run (run_to_dict's shape) with the given per-hour values."""
    starts = [first + timedelta(hours=i) for i in range(len(rows))]
    return {
        "computed_at": computed_at.isoformat(),
        "inputs": {
            "hour_starts": [s.isoformat() for s in starts],
            "t_out_c": [r["t_out_c"] for r in rows],
            "sun_wm2": [r["sun_wm2"] for r in rows],
        },
        "decided": [
            {
                "start": s.isoformat(),
                "rts_c": r["rts_c"],
                "energy_kwh": r["energy_kwh"],
                "room_c": r.get("room_c"),
                "flow_c": r.get("flow_c"),
            }
            for s, r in zip(starts, rows, strict=True)
        ],
    }


ROW = {
    "rts_c": 16.0,
    "energy_kwh": 1.0,
    "room_c": 20.0,
    "flow_c": 30.0,
    "t_out_c": 5.0,
    "sun_wm2": 0.0,
}
MEASURED = {
    "rts_c": 16.0,
    "energy_kwh": 1.5,
    "room_c": 19.5,
    "flow_c": 31.0,
    "t_out_c": 4.0,
    "sun_wm2": 10.0,
}


def test_an_hour_is_compared_with_what_each_plan_said_by_how_far_ahead_it_was_planned():
    runs = [
        run(HOUR - timedelta(minutes=10), HOUR, [ROW]),  # ends 70 min later: 1-3 h
        run(HOUR - timedelta(hours=8), HOUR - timedelta(hours=8), [ROW] * 9),
    ]
    line = check_hour(HOUR, runs, MEASURED)
    assert line["hour"] == HOUR.isoformat()
    assert line["measured"] == MEASURED
    assert set(line["predicted"]) == {"1-3 h", "6-12 h"}
    assert line["predicted"]["6-12 h"]["room_c"] == 20.0
    assert line["runs"] == {"1-3 h": 1, "6-12 h": 1}


def test_plans_in_the_same_lead_are_averaged():
    late = dict(ROW, room_c=21.0)
    runs = [
        run(HOUR - timedelta(hours=2), HOUR - timedelta(hours=2), [ROW] * 3),
        run(
            HOUR - timedelta(hours=1, minutes=30), HOUR - timedelta(hours=2), [late] * 3
        ),
    ]
    assert check_hour(HOUR, runs, MEASURED)["predicted"]["1-3 h"]["room_c"] == 20.5


def test_a_run_from_before_room_and_flow_were_kept_still_counts_for_the_rest():
    old = {k: v for k, v in ROW.items() if k not in ("room_c", "flow_c")}
    line = check_hour(HOUR, [run(HOUR - timedelta(hours=2), HOUR, [old])], MEASURED)
    assert line["predicted"]["1-3 h"] == {
        "rts_c": 16.0,
        "energy_kwh": 1.0,
        "t_out_c": 5.0,
        "sun_wm2": 0.0,
    }


def test_a_run_that_did_not_plan_the_hour_is_left_out():
    later = run(HOUR + timedelta(hours=1), HOUR + timedelta(hours=1), [ROW])
    assert check_hour(HOUR, [later], MEASURED)["predicted"] == {}


def test_the_summary_gives_bias_and_mean_error_per_quantity_and_lead():
    lines = [
        check_hour(HOUR, [run(HOUR - timedelta(hours=2), HOUR, [ROW])], MEASURED),
        check_hour(
            HOUR + timedelta(hours=1),
            [run(HOUR - timedelta(hours=1), HOUR + timedelta(hours=1), [ROW])],
            dict(MEASURED, room_c=20.5),
        ),
    ]
    rows = {(r["quantity"], r["lead"]): r for r in summarize(lines)["rows"]}
    room = rows[("room_c", "1-3 h")]
    assert room["n"] == 2
    assert room["bias"] == pytest.approx(0.0)  # +0.5 and -0.5
    assert room["mae"] == pytest.approx(0.5)


def test_hours_without_a_measurement_are_counted_not_scored():
    line = check_hour(
        HOUR,
        [run(HOUR - timedelta(hours=2), HOUR, [ROW])],
        dict(MEASURED, room_c=None),
    )
    summary = summarize([line])
    assert summary["missing"]["room_c"] == 1
    assert ("room_c", "1-3 h") not in {
        (r["quantity"], r["lead"]) for r in summary["rows"]
    }


def measured_for(start, stop):
    hours = [
        start + timedelta(hours=i)
        for i in range(int((stop - start) / timedelta(hours=1)))
    ]
    full = np.full(len(hours), 1.0)
    return Measured(hours, *(full.copy() for _ in range(6)))


class FakeRuns:
    def __init__(self, runs):
        self.runs = runs

    def read(self, since):
        return [
            r for r in self.runs if datetime.fromisoformat(r["computed_at"]) >= since
        ]


def test_every_finished_hour_is_checked_once(tmp_path):
    now = [HOUR + timedelta(minutes=5)]
    store = ForecastCheckStore(str(tmp_path / "check.jsonl"))
    runs = FakeRuns(
        [run(HOUR - timedelta(hours=6), HOUR - timedelta(hours=6), [ROW] * 6)]
    )
    checker = ForecastChecker(runs, measured_for, store, lambda: now[0])
    checker.run_due()
    hours = [line["hour"] for line in store.read(HOUR - timedelta(days=30))]
    assert (
        hours[-1] == (HOUR - timedelta(hours=2)).isoformat()
    )  # an hour that ended an hour ago
    checker.run_due()
    assert len(store.read(HOUR - timedelta(days=30))) == len(hours)
    now[0] += timedelta(hours=1)
    checker.run_due()
    assert (
        store.read(HOUR - timedelta(days=30))[-1]["hour"]
        == (HOUR - timedelta(hours=1)).isoformat()
    )


def test_the_check_goes_back_no_further_than_the_plans_are_kept(tmp_path):
    store = ForecastCheckStore(str(tmp_path / "check.jsonl"))
    checker = ForecastChecker(FakeRuns([]), measured_for, store, lambda: HOUR)
    checker.run_due()
    first = datetime.fromisoformat(store.read(HOUR - timedelta(days=30))[0]["hour"])
    assert first >= HOUR - timedelta(days=6)


def test_the_store_survives_a_restart(tmp_path):
    path = str(tmp_path / "check.jsonl")
    ForecastChecker(
        FakeRuns([]), measured_for, ForecastCheckStore(path), lambda: HOUR
    ).run_due()
    again = ForecastCheckStore(path)
    assert again.last_hour() == HOUR - timedelta(hours=2)


class FakeSource:
    def __init__(self, points):
        self.points = points

    def series(self, entity_id, start, stop, field=None):
        return self.points.get((entity_id, field), [])


def test_the_room_is_measured_at_the_end_of_the_hour(settings):
    t = HOUR
    source = FakeSource(
        {
            (settings.room_temp_entities[0], None): [
                (t + timedelta(minutes=5), 19.0),
                (t + timedelta(minutes=55), 20.0),
            ],
        }
    )
    measured = load_measured(
        source, settings, lambda a, b: [], t, t + timedelta(hours=1)
    )
    assert measured.at(t)["room_c"] == 20.0
    assert measured.at(t)["energy_kwh"] is None  # no counter: not measured


def test_the_flow_target_holds_between_reports(settings):
    """The pump's flow target is logged only when it changes."""
    source = FakeSource(
        {(settings.flow_setpoint_entity, None): [(HOUR + timedelta(minutes=30), 35.0)]}
    )
    measured = load_measured(
        source, settings, lambda a, b: [], HOUR, HOUR + timedelta(hours=4)
    )
    assert measured.at(HOUR + timedelta(hours=3))["flow_c"] == 35.0
