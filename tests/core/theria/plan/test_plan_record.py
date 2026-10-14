from datetime import UTC, datetime, timedelta

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.plan_record import PlanRecord, PlanRecordStore
from core.theria.plan.series import hourly_grid

from .test_day_view import LOCAL_DAY, PLAN_START
from .test_service_active import make, run_cycle

HOUR = datetime(2026, 10, 12, 14, tzinfo=UTC)


def record(hour=HOUR, rts=17.0):
    return PlanRecord(
        hour=hour,
        computed_at=hour + timedelta(minutes=3),
        rts_c=rts,
        room_c=19.4,
        energy_kwh=1.5,
        baseline_rts_c=18.0,
        baseline_room_c=19.5,
        baseline_energy_kwh=2.0,
        price_sek_per_kwh=1.2,
    )


def test_a_record_is_read_back(tmp_path):
    store = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    store.append(record())
    again = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    found = again.read(HOUR - timedelta(hours=1), HOUR + timedelta(hours=1))
    assert found[HOUR].rts_c == 17.0 and found[HOUR].baseline_energy_kwh == 2.0


def test_only_the_first_plan_of_an_hour_is_kept_it_is_the_one_that_was_applied(tmp_path):
    store = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    store.append(record(rts=17.0))
    store.append(record(rts=19.0))
    assert store.read(HOUR, HOUR + timedelta(hours=1))[HOUR].rts_c == 17.0


def test_only_the_asked_range_is_returned(tmp_path):
    store = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    for k in range(5):
        store.append(record(hour=HOUR + timedelta(hours=k)))
    found = store.read(HOUR + timedelta(hours=1), HOUR + timedelta(hours=3))
    assert sorted(found) == [HOUR + timedelta(hours=1), HOUR + timedelta(hours=2)]


def test_a_damaged_line_is_skipped(tmp_path):
    path = tmp_path / "plan_record.jsonl"
    store = PlanRecordStore(str(path))
    store.append(record())
    with open(path, "a") as f:
        f.write("{not json\n")
    assert HOUR in PlanRecordStore(str(path)).read(HOUR, HOUR + timedelta(hours=1))


# ---- in the service and the day view


def measured_with_energy(start, stop):
    n = int((stop - start).total_seconds() // 3600)
    nothing = np.full(n, np.nan)
    return OutcomeInputs(
        hours=hourly_grid(start, n),
        price_sek_per_kwh=nothing,
        rts_c=np.full(n, 18.0),
        t_out_c=nothing,
        compressor_kwh=np.full(n, 1.0),
        aux_kwh=np.full(n, 0.5),
        hot_water=np.zeros(n),
        room_c=np.full(n, 20.0),
    )


def test_each_hour_the_plan_that_was_applied_is_recorded(settings, tmp_path):
    store = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    service, *_ = make(settings, plan_record=store)
    run_cycle(service)
    found = store.read(PLAN_START, PLAN_START + timedelta(hours=1))
    first = service.latest().plan.hours[0]
    assert found[PLAN_START].rts_c == first.rts_c
    run_cycle(service)  # the same hour again: nothing new
    assert len(store.read(PLAN_START, PLAN_START + timedelta(hours=1))) == 1


def test_a_past_hour_shows_the_plan_that_was_made_for_it_next_to_what_happened(
    settings, tmp_path
):
    store = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    past_hour = PLAN_START - timedelta(hours=2)
    store.append(record(hour=past_hour, rts=17.0))
    service, *_ = make(settings, plan_record=store, actuals_loader=measured_with_energy)
    run_cycle(service)
    rows = {r["start"]: r for r in service.day_view(LOCAL_DAY)["hours"]}
    row = rows[past_hour]
    assert row["past"]
    assert row["rts_measured"] == 18.0 and row["rts_planned"] == 17.0
    assert row["rts_baseline"] == 18.0
    assert row["room_measured"] == 20.0 and row["room_planned"] == 19.4
    assert row["energy_measured_kwh"] == 1.5 and row["energy_planned_kwh"] == 1.5
    assert row["energy_baseline_kwh"] == 2.0


def test_the_saving_per_hour_is_the_baseline_energy_minus_the_energy_at_the_hours_price(
    settings, tmp_path
):
    store = PlanRecordStore(str(tmp_path / "plan_record.jsonl"))
    past_hour = PLAN_START - timedelta(hours=2)
    store.append(record(hour=past_hour))  # planned 1.5 kWh against 2.0 kWh flat, at 1.2 kr
    service, *_ = make(settings, plan_record=store, actuals_loader=measured_with_energy)
    run_cycle(service)
    row = {r["start"]: r for r in service.day_view(LOCAL_DAY)["hours"]}[past_hour]
    assert row["saving_planned_sek"] == (2.0 - 1.5) * 1.2
    assert row["saving_measured_sek"] == (2.0 - 1.5) * 1.2  # 1.0 + 0.5 kWh measured


def test_a_future_hour_has_plan_and_baseline_but_nothing_measured(settings):
    service, *_ = make(settings)
    run_cycle(service)
    row = {r["start"]: r for r in service.day_view(LOCAL_DAY)["hours"]}[PLAN_START]
    assert row["rts_planned"] is not None and row["rts_baseline"] is not None
    assert row["rts_measured"] is None and row["energy_measured_kwh"] is None
    assert row["saving_measured_sek"] is None
