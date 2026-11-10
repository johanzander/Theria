from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.day_view import day_hours
from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.room_sections import SectionMeasured
from core.theria.plan.series import hourly_grid
from core.theria.plan.zone_heat import (
    ZoneMeasured,
    delivered_ground_kw,
    expected_open_share,
    ground_need_kw,
    load_zone_measurements,
    salon_need_kw,
)

from .test_day_view import LOCAL_DAY
from .test_service_active import make, run_cycle

START = datetime(2026, 10, 12, 0, tzinfo=UTC)


def test_the_ground_floor_need_is_its_loss_to_the_outdoors():
    assert ground_need_kw(19.5, 9.5) == pytest.approx(1.615)


def test_a_warm_house_with_closed_loops_still_needs_heat():
    assert ground_need_kw(21.0, 6.5) == pytest.approx(2.342, abs=0.001)


def test_the_salon_need_is_the_loss_minus_the_measured_other_heat():
    assert salon_need_kw(20.0, 10.0) == pytest.approx(0.842 - 0.25)


def test_the_salon_never_needs_less_than_nothing():
    assert salon_need_kw(16.0, 15.0) == 0.0


def test_open_loops_deliver_by_the_flow_above_the_room():
    assert delivered_ground_kw(35.0, 20.0, 1.0, 1.0, 40.0) == pytest.approx(2.505)


def test_closed_loops_deliver_nothing_whatever_the_flow():
    assert delivered_ground_kw(45.0, 20.0, 0.0, 1.0, 40.0) == 0.0


def test_the_floor_gets_at_most_what_the_shunts_let_through():
    assert delivered_ground_kw(50.0, 20.0, 1.0, 1.0, 40.0) == pytest.approx(3.34)


def test_time_spent_on_hot_water_delivers_nothing_to_the_floor():
    assert delivered_ground_kw(35.0, 20.0, 1.0, 0.5, 40.0) == pytest.approx(1.2525)


def test_delivery_is_unknown_without_a_flow():
    assert delivered_ground_kw(float("nan"), 20.0, 1.0, 1.0, 40.0) is None


def test_an_unknown_input_gives_no_need():
    assert ground_need_kw(19.5, float("nan")) is None
    assert salon_need_kw(float("nan"), 5.0) is None


@pytest.fixture
def shift(settings):
    return replace(settings)


def test_a_closing_thermostat_expects_closed_loops_above_its_target(shift):
    assert expected_open_share("closing", 19.9, False, shift) == 0.0


def test_a_closing_thermostat_reopens_a_room_that_has_fallen(shift):
    assert expected_open_share("closing", 18.0, False, shift) == 1.0


def test_an_opening_thermostat_expects_open_loops(shift):
    assert expected_open_share("opening", 19.5, False, shift) == 1.0


def test_a_normal_thermostat_opens_below_the_normal_target(shift):
    assert expected_open_share("normal", 19.25, False, shift) == pytest.approx(
        1.0 / 1.7
    )


class Source:
    def __init__(self, series):
        self.data = series

    def series(self, entity, start, stop, field=None):
        values = self.data.get((entity, field), [])
        return [(start + timedelta(hours=i), float(v)) for i, v in enumerate(values)]


def test_the_salon_is_the_mean_of_its_points(settings):
    source = Source(
        {
            ("climate.vantsidan", "current_temperature"): [20.0, 21.0],
            ("sensor.floor", None): [18.0, 19.0],
            (settings.flow_measured_entity, None): [35.0, 36.0],
            (settings.heating_mode_entity, None): [0, 0],
        }
    )
    zones = load_zone_measurements(
        source,
        settings,
        (("climate.vantsidan", "current_temperature"), ("sensor.floor", None)),
        (),
        START,
        START + timedelta(hours=2),
    )
    assert zones.salon_c == pytest.approx([19.0, 20.0])
    assert zones.flow_c == pytest.approx([35.0, 36.0])


class TimedSource:
    def __init__(self, series):
        self.data = series

    def series(self, entity, start, stop, field=None):
        return [
            (start + timedelta(minutes=m), float(v))
            for m, v in self.data.get(entity, [])
        ]


def flow_with_valve(settings, flow, valve, hours=2):
    source = TimedSource(
        {settings.flow_measured_entity: flow, settings.heating_mode_entity: valve}
    )
    return load_zone_measurements(
        source, settings, (), (), START, START + timedelta(hours=hours)
    )


def test_the_flow_leaves_out_the_hot_water_and_the_quarter_after_it(settings):
    flow = flow_with_valve(
        settings,
        [(5, 36.0), (20, 55.0), (35, 30.0), (50, 38.0)],
        [(0, 0), (15, 1), (25, 0)],
        hours=1,
    ).flow_c
    assert flow[0] == pytest.approx(37.0)  # 36 before, 38 after; 55 and 30 are left out


def test_an_hour_of_only_hot_water_has_no_flow(settings):
    flow = flow_with_valve(settings, [(10, 50.0), (70, 37.0)], [(0, 1), (55, 0)]).flow_c
    assert np.isnan(flow[0]) and flow[1] == pytest.approx(37.0)


def test_the_pumps_own_target_is_kept_between_reports(settings):
    source = TimedSource(
        {settings.flow_setpoint_entity: [(10, 38.0), (130, 36.0), (140, 0.0)]}
    )
    target = load_zone_measurements(
        source, settings, (), (), START, START + timedelta(hours=3)
    ).flow_target_c
    assert target == pytest.approx([38.0, 38.0, 36.0])


# On the dashboard


def measured(start, stop):
    n = int((stop - start).total_seconds() // 3600)
    nothing = np.full(n, np.nan)
    return OutcomeInputs(
        hours=hourly_grid(start, n),
        price_sek_per_kwh=nothing,
        rts_c=np.full(n, 18.0),
        t_out_c=np.full(n, 5.0),
        compressor_kwh=nothing,
        aux_kwh=nothing,
        hot_water=nothing,
        room_c=np.full(n, 19.2),
        thermostat_set_c={"lk_bibliotek": np.full(n, 19.5)},
    )


def zones(start, stop):
    n = int((stop - start).total_seconds() // 3600)
    return ZoneMeasured(
        hourly_grid(start, n),
        np.full(n, 37.0),
        np.full(n, 19.8),
        np.full(n, 38.0),
        np.full(n, 6.0),
    )


def shifting(settings):
    return replace(settings, rts_base_c=18.0, plan_closes_valves=True)


def view(settings):
    service, *_ = make(shifting(settings), actuals_loader=measured, zone_loader=zones)
    service.use_salon_status(lambda: _salon(20.5))
    run_cycle(service)
    return service.day_view(LOCAL_DAY)["hours"]


def _salon(mean):
    from core.theria.plan.shift_rules import SalonStatus

    return SalonStatus(mean, 17.0)


def test_a_past_hour_shows_the_salon_and_the_flow_as_measured(settings):
    past = next(h for h in view(settings) if h["past"])
    assert past["salon_measured"] == 19.8
    assert past["flow_measured_c"] == 37.0
    assert past["flow_target_c"] == 38.0
    assert past["need_salon_kw"] == pytest.approx(salon_need_kw(19.8, 5.0))
    assert past["need_ground_kw"] == pytest.approx(ground_need_kw(19.2, 5.0))
    # no actuators are configured in this test, so nothing is known to be open
    assert past["delivered_ground_kw"] is None
    assert past["thermostat_target_c"] == 19.5


def test_a_past_hour_shows_each_sections_rooms_and_thermostats(settings):
    def with_sections(start, stop):
        base = zones(start, stop)
        n = len(base.hours)
        guest = SectionMeasured(
            rooms={"Kök": np.full(n, 20.1)},
            setpoints={"Kök": np.full(n, 20.0)},
            demand={"Kök": np.full(n, np.nan)},
        )
        return replace(base, sections={"guest": guest})

    service, *_ = make(
        shifting(settings), actuals_loader=measured, zone_loader=with_sections
    )
    run_cycle(service)
    hours = service.day_view(LOCAL_DAY)["hours"]
    past = next(h for h in hours if h["past"])
    assert past["sections"] == {
        "guest": {
            "rooms": {"Kök": 20.1},
            "setpoints": {"Kök": 20.0},
            "demand": {"Kök": None},
        }
    }
    assert next(h for h in hours if not h["past"])["sections"] == {}


def test_the_whole_house_is_shown_from_its_fit_and_the_heat_counter(settings):
    from core.theria.plan.day_view import build_day_view
    from core.theria.plan.house_loss import HouseLoss

    house = HouseLoss(w_per_k=700.0, free_kw=3.0, days=160, rmse_kw=1.1)
    first = day_hours(LOCAL_DAY, settings.timezone)[0]
    now = first + timedelta(hours=6)
    view = build_day_view(
        LOCAL_DAY,
        now,
        settings,
        [],
        None,
        measured(first, now),
        zones=zones(first, now),
        house=house,
    )
    past = view["hours"][0]
    assert past["need_house_kw"] == pytest.approx(0.7 * (19.2 - 5.0))
    assert past["house_pump_kw"] == pytest.approx(0.7 * (19.2 - 5.0) - 3.0)
    assert past["heat_measured_kw"] == 6.0
    assert view["house"]["w_per_k"] == 700.0
    assert build_day_view(LOCAL_DAY, now, settings, [], None, None)["house"] is None


def test_a_coming_hour_shows_what_the_plan_expects(settings):
    coming = next(h for h in view(settings) if not h["past"] and h["rts_planned"])
    assert coming["salon_planned"] == pytest.approx(20.5, abs=0.5)
    assert coming["need_ground_kw"] is not None
    assert coming["delivered_ground_kw"] is not None
    assert coming["thermostat"] in ("closing", "opening", "normal")
    assert 0.0 <= coming["loops_expected"] <= 1.0
    assert coming["thermostat_target_c"] is not None


def test_a_past_hour_delivers_by_the_loops_that_were_open(settings):
    def with_loops(start, stop):
        base = measured(start, stop)
        n = len(base.hours)
        return replace(
            base,
            hot_water=np.zeros(n),
            loops_open={"vardagsrum": np.full(n, 3.0), "bibliotek": np.zeros(n)},
        )

    actuators = {"vardagsrum": ("a", "b", "c"), "bibliotek": ("d", "e")}
    service, *_ = make(
        replace(shifting(settings), room_actuators=actuators),
        actuals_loader=with_loops,
        zone_loader=zones,
    )
    run_cycle(service)
    past = next(h for h in service.day_view(LOCAL_DAY)["hours"] if h["past"])
    assert past["delivered_ground_kw"] == pytest.approx(0.078 * (37.0 - 19.2))
