from datetime import UTC, date, datetime, timedelta

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.series import hourly_grid

from .test_service_active import make, run_cycle

LOCAL_DAY = date(2026, 10, 12)
FIRST_HOUR = datetime(2026, 10, 11, 22, tzinfo=UTC)  # local midnight (UTC+2)
PLAN_START = datetime(2026, 10, 12, 16, tzinfo=UTC)


def measured(start, stop):
    n = int((stop - start).total_seconds() // 3600)
    nothing = np.full(n, np.nan)
    return OutcomeInputs(
        hours=hourly_grid(start, n),
        price_sek_per_kwh=nothing,
        rts_c=np.full(n, 18.0),
        t_out_c=nothing,
        compressor_kwh=nothing,
        aux_kwh=nothing,
        hot_water=nothing,
        room_c=np.full(n, 20.0),
    )


def started(settings, actuals=measured):
    service, *_ = make(settings, actuals_loader=actuals)
    run_cycle(service)
    return service


def test_the_day_is_always_24_local_hours_from_midnight(settings):
    rows = started(settings).day_view(LOCAL_DAY)["hours"]
    assert len(rows) == 24
    assert rows[0]["start"] == FIRST_HOUR and rows[-1][
        "start"
    ] == FIRST_HOUR + timedelta(hours=23)


def test_hours_already_passed_show_what_was_measured_and_the_rest_the_plan(settings):
    service = started(settings)
    rows = {r["start"]: r for r in service.day_view(LOCAL_DAY)["hours"]}
    past = rows[PLAN_START - timedelta(hours=1)]
    assert past["past"] and past["rts_c"] == 18.0 and past["room_c"] == 20.0
    assert past["baseline_rts_c"] is None
    planned = rows[PLAN_START]
    first = service.latest().plan.hours[0]
    assert not planned["past"] and planned["rts_c"] == first.rts_c
    assert (
        planned["room_c"] == first.room_c_end and planned["baseline_rts_c"] is not None
    )


def test_prices_cover_the_hours_that_have_passed_too(settings):
    rows = {r["start"]: r for r in started(settings).day_view(LOCAL_DAY)["hours"]}
    assert rows[PLAN_START - timedelta(hours=3)]["price_sek_per_kwh"] == 1.0
    assert rows[PLAN_START]["price_sek_per_kwh"] == 3.0


def test_tomorrow_is_all_plan(settings):
    view = started(settings).day_view(LOCAL_DAY + timedelta(days=1))
    assert len(view["hours"]) == 24 and not any(r["past"] for r in view["hours"])


def test_the_chart_still_has_prices_when_the_measurements_cannot_be_read(settings):
    def broken(start, stop):
        raise ModelFitError("InfluxDB is down")

    rows = {
        r["start"]: r for r in started(settings, broken).day_view(LOCAL_DAY)["hours"]
    }
    past = rows[PLAN_START - timedelta(hours=1)]
    assert past["rts_c"] is None and past["price_sek_per_kwh"] == 1.0


def test_hours_that_passed_carry_the_open_floor_loops_per_room(settings):
    from dataclasses import replace

    settings = replace(settings, room_actuators={"bib": ("a", "b", "c")})

    def with_loops(start, stop):
        base = measured(start, stop)
        return replace(base, loops_open={"bib": np.full(len(base.hours), 2.0)})

    view = started(settings, with_loops).day_view(LOCAL_DAY)
    rows = {r["start"]: r for r in view["hours"]}
    assert view["loops_total"] == {"bib": 3}
    assert rows[PLAN_START - timedelta(hours=1)]["loops_open"] == {"bib": 2.0}
    assert rows[PLAN_START]["loops_open"] == {}


def test_hours_that_passed_carry_each_thermostats_temperature_and_setpoint(settings):
    from dataclasses import replace

    def with_thermostats(start, stop):
        base = measured(start, stop)
        n = len(base.hours)
        return replace(
            base,
            thermostat_room_c={"lk_a": np.full(n, 19.8)},
            thermostat_set_c={"lk_a": np.full(n, 19.5)},
        )

    rows = {
        r["start"]: r
        for r in started(settings, with_thermostats).day_view(LOCAL_DAY)["hours"]
    }
    assert rows[PLAN_START - timedelta(hours=1)]["thermostats"] == {
        "lk_a": {"room_c": 19.8, "set_c": 19.5}
    }
    assert rows[PLAN_START]["thermostats"] == {}


def test_every_hour_carries_its_four_quarter_prices_and_the_hour_price_is_their_mean(
    settings,
):
    rows = started(settings).day_view(LOCAL_DAY)["hours"]
    priced = [r for r in rows if r["price_sek_per_kwh"] is not None]
    assert priced
    for row in rows:
        assert len(row["price_quarters"]) == 4
    for row in priced:
        assert row["price_sek_per_kwh"] == np.mean(row["price_quarters"])
    for row in rows:
        if row["price_sek_per_kwh"] is None:
            assert None in row["price_quarters"]
