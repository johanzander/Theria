from dataclasses import replace

import numpy as np
import pytest

from core.theria.exceptions import OptimizationError
from core.theria.plan.planner import TAIL_HOURS, PlanInputs, plan
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, flow_target
from core.theria.plan.room_model import (
    PRIOR_ROOM_PARAMS,
    RoomState,
    anchored_room_params,
    step,
)

from .conftest import HORIZON

BASE_SETTINGS = {
    "price_entity": "sensor.price",
    "outdoor_entity": "sensor.outdoor",
    "room_setpoint_entity": "climate.room_temp_setpoint",
    "flow_setpoint_entity": "sensor.heating_setpoint",
    "room_temp_entities": ["sensor.a"],
    "room_thermostat_entities": ["climate.a"],
    "room_history_entities": [],
    "heating_mode_entity": "binary_sensor.valve",
    "flow_measured_entity": "sensor.flow",
    "energy_counter_entity": "sensor.energy",
    "aux_energy_entity": "sensor.aux",
    "aux_status_entity": "sensor.aux_status",
    "latitude": 55.5,
    "longitude": 12.9,
}

HOUR_OF_DAY = np.arange(HORIZON) % 24
EVENING = (HOUR_OF_DAY >= 16) & (HOUR_OF_DAY < 20)


def run(inputs, settings):
    """Plan the way the service does: the baseline holds the room where it is."""
    baseline_flow = np.minimum(
        flow_target(PRIOR_PUMP_PARAMS, inputs.t_out_c, settings.rts_base_c),
        settings.floor_mix_max_c,
    )
    room = anchored_room_params(
        PRIOR_ROOM_PARAMS,
        inputs.initial.room_c,
        inputs.t_out_c,
        baseline_flow,
        settings.lk_limit_c,
    )
    return plan(inputs, settings, PRIOR_PUMP_PARAMS, room)


def test_flat_price_plan_respects_the_rise_limit_and_the_setpoint_range(
    settings, make_inputs
):
    result = run(make_inputs(np.full(HORIZON, 1.0)), settings)
    rts = [h.rts_c for h in result.hours]
    assert all(settings.rts_min_c <= r <= settings.rts_max_c for r in rts)
    rises = np.diff([settings.rts_base_c, *rts])
    assert (
        rises.max() <= settings.max_rts_step_per_hour_c + 1e-9
    )  # lowering is not limited


def test_price_spike_lowers_setpoint_during_spike_and_cuts_cost(settings, make_inputs):
    price = np.where(EVENING, 4.0, 0.8)
    result = run(make_inputs(price), settings)
    during = np.mean([h.rts_c for h, e in zip(result.hours, EVENING, strict=True) if e])
    outside = np.mean(
        [h.rts_c for h, e in zip(result.hours, EVENING, strict=True) if not e]
    )
    assert during < outside - 1.0
    assert result.objective_sek < result.baseline_objective_sek


def test_every_hour_has_a_reason_and_last_hours_are_flagged_as_tail(
    settings, make_inputs
):
    result = run(make_inputs(np.where(EVENING, 4.0, 0.8)), settings)
    assert all("binding" in h.reason and "price_rank" in h.reason for h in result.hours)
    assert [h.tail for h in result.hours].count(True) == TAIL_HOURS
    assert result.hours[-1].tail and not result.hours[0].tail


def test_floor_is_held_when_it_is_reachable(settings, make_inputs):
    price = np.where(EVENING, 6.0, 0.7)
    result = run(make_inputs(price, t_out=4.0, room_c=19.5, slab_c=30.0), settings)
    assert result.floor_violation_hours == 0
    assert min(h.room_c_end for h in result.hours) >= settings.floor_c - 0.05


def test_unreachable_floor_is_reported_not_raised(settings, make_inputs):
    result = run(
        make_inputs(np.full(HORIZON, 1.0), t_out=-5.0, room_c=17.0, slab_c=20.0),
        settings,
    )
    assert result.floor_violation_hours > 0


def test_extreme_hours_use_the_lower_floor(settings, make_inputs):
    price = np.where(EVENING, 8.0, 0.8)
    result = run(make_inputs(price, extreme=EVENING, t_out=2.0, slab_c=33.0), settings)
    for hour, is_extreme in zip(result.hours, EVENING, strict=True):
        assert hour.floor_c == (
            settings.extreme_floor_c if is_extreme else settings.floor_c
        )


def test_too_short_horizon_raises(settings, make_inputs):
    inputs = make_inputs(np.full(HORIZON, 1.0))
    short = type(inputs)(
        hour_starts=inputs.hour_starts[:1],
        price_sek_per_kwh=inputs.price_sek_per_kwh[:1],
        extreme=inputs.extreme[:1],
        t_out_c=inputs.t_out_c[:1],
        sun_wm2=inputs.sun_wm2[:1],
        initial=inputs.initial,
        previous_rts_c=inputs.previous_rts_c,
    )
    with pytest.raises(OptimizationError):
        run(short, settings)


def test_negative_prices_never_push_the_room_above_the_ceiling(settings, make_inputs):
    price = np.where(EVENING, -0.8, 1.0)
    result = run(make_inputs(price), settings)
    assert max(h.room_c_end for h in result.hours) <= settings.ceiling_c + 0.05
    during = np.mean([h.rts_c for h, e in zip(result.hours, EVENING, strict=True) if e])
    assert during > settings.rts_base_c  # being paid to heat is used


def test_cold_snap_keeps_heating_at_maximum_instead_of_giving_up(settings, make_inputs):
    price = np.where(EVENING, 6.0, 0.7)
    result = run(make_inputs(price, t_out=-10.0, room_c=19.5, slab_c=40.0), settings)
    # hours 0-2 ramp up from the current setpoint within the rate limit; after that the
    # plan heats as hard as the flow-temperature cap allows
    hot = result.hours[3:12]
    assert all(h.flow_target_c > settings.max_flow_c - 1.0 for h in hot)


def test_flow_temperature_never_exceeds_the_floor_limit(settings, make_inputs):
    price = np.where(EVENING, 6.0, 0.7)
    result = run(make_inputs(price, t_out=-10.0, room_c=19.5, slab_c=40.0), settings)
    assert max(h.flow_target_c for h in result.hours) <= settings.max_flow_c
    capped = [h for h in result.hours if h.reason["binding"] == "flow_cap"]
    assert capped, "the cap should be what limits the plan in a cold snap"


def test_replanning_from_the_measured_room_does_not_ratchet_the_room_down(settings):
    """Every replan re-anchors the baseline at the room's current temperature. A sag caused by
    earlier plans must not become the next plan's equilibrium: the deficit lasts about the
    room's time constant (days), not the few hours of the horizon."""
    from datetime import UTC, datetime, timedelta

    pump = replace(PRIOR_PUMP_PARAMS, energy_per_rts=0.223)  # the fitted value
    horizon, replan_every, days = 30, 12, 6
    t_out, lk = 5.0, settings.lk_limit_c
    base_flow = flow_target(pump, np.full(horizon, t_out), settings.rts_base_c)
    # the real house: holds 20.3 C on the baseline setpoint
    true = anchored_room_params(
        PRIOR_ROOM_PARAMS, 20.3, np.full(horizon, t_out), base_flow, lk
    )
    room, slab = 20.3, float(base_flow[0])
    previous_rts = settings.rts_base_c
    history = [room]
    start = datetime(2026, 10, 12, tzinfo=UTC)
    for replan in range(days * 24 // replan_every):
        now_hour = replan * replan_every
        hours = [start + timedelta(hours=now_hour + h) for h in range(horizon)]
        evening = np.array(
            [(now_hour + h) % 24 in (16, 17, 18, 19) for h in range(horizon)]
        )
        price = np.where(evening, 3.5, 1.8)
        anchored = anchored_room_params(
            PRIOR_ROOM_PARAMS, room, np.full(horizon, t_out), base_flow, lk
        )
        inputs = PlanInputs(
            hour_starts=hours,
            price_sek_per_kwh=price,
            extreme=np.zeros(horizon, bool),
            t_out_c=np.full(horizon, t_out),
            sun_wm2=np.zeros(horizon),
            initial=RoomState(room, slab, 0.0),
            previous_rts_c=previous_rts,
        )
        result = plan(inputs, settings, pump, anchored)
        for hour in result.hours[:replan_every]:
            flow = float(flow_target(pump, t_out, hour.rts_c))
            room, slab, _ = step(true, room, slab, 0.0, flow, t_out, 0.0, lk)
            history.append(float(room))
            previous_rts = hour.rts_c
    assert min(history) > settings.target_c - 0.4
    assert history[-1] > settings.target_c - 0.4


def daily_prices(cheap_night=0.9):
    """A typical Swedish winter day: cheap night, morning peak, cheap midday, long evening peak."""
    hour = np.arange(36) % 24
    return np.select(
        [
            (hour >= 6) & (hour < 9),
            (hour >= 16) & (hour < 21),
            (hour >= 0) & (hour < 5),
        ],
        [2.6, 3.4, cheap_night],
        default=1.6,
    )


def test_the_setpoint_varies_over_the_day_and_the_average_stays_near_the_base(
    settings, make_inputs
):
    result = run(make_inputs(daily_prices(cheap_night=0.3), room_c=20.0, slab_c=33.0), settings)
    rts = np.array([h.rts_c for h in result.hours])
    assert rts.min() < settings.rts_base_c - 0.5  # lower in the expensive hours
    assert rts.max() >= settings.rts_base_c + 0.5  # higher in the cheap hours
    for first in range(0, 12):
        window = rts[first : first + 24]
        assert (
            abs(window.mean() - settings.rts_base_c) < 1.0
        ), "a day must be energy neutral"


def test_the_heat_is_never_off_for_long_stretches(settings, make_inputs):
    result = run(make_inputs(daily_prices(), room_c=20.0, slab_c=33.0), settings)
    longest, current = 0, 0
    for h in result.hours:
        current = current + 1 if h.rts_c <= settings.rts_base_c - 2.0 else 0
        longest = max(longest, current)
    assert longest <= 6


def test_the_room_ends_the_horizon_near_where_the_flat_setpoint_would_leave_it(
    settings, make_inputs
):
    result = run(make_inputs(daily_prices(), room_c=20.3, slab_c=33.0), settings)
    flat_end = result.baseline_hours[-1].room_c_end
    assert result.hours[-1].room_c_end > flat_end - 0.15


def test_a_room_the_flat_setpoint_keeps_below_target_still_shifts_heat_to_the_cheap_hours(
    settings, make_inputs
):
    # The flat setpoint holds this room at about 19.0, under the 19.5 target. The end of the
    # horizon must be judged against where the flat setpoint ends, not against a target it never
    # reaches: otherwise every degree at the end outweighs the price and the plan just heats.
    result = run(make_inputs(daily_prices(), room_c=19.0, slab_c=28.0), settings)
    assert result.objective_sek < result.baseline_objective_sek - 0.1
    rts = np.array([h.rts_c for h in result.hours])
    price = daily_prices()
    assert rts[price == price.min()].mean() > rts[price == price.max()].mean() + 0.5


def test_an_expensive_peak_gets_less_heat_than_the_hours_before_it(
    settings, make_inputs
):
    # The cut lowers the room by hundredths of a degree; what that costs is the energy it takes
    # to bring those hundredths back at an ordinary price, not a comfort deficit that is held
    # against the plan for days. Heat is bought before the peak instead and the peak gets less.
    price = np.full(HORIZON, 1.0)
    price[8:16] = 2.5
    result = run(make_inputs(price, room_c=19.0, slab_c=28.0), settings)
    rts = [h.rts_c for h in result.hours]
    assert np.mean(rts[8:16]) < np.mean(rts[:8]) - 0.5
    assert result.baseline_objective_sek - result.objective_sek > 1.0


def _end_cost_per_degree(settings, make_inputs, pump):
    from core.theria.plan.planner import _Problem

    inputs = make_inputs(np.full(HORIZON, 1.0), room_c=19.0, slab_c=28.0)
    problem = _Problem(inputs, settings, pump, PRIOR_ROOM_PARAMS)
    return problem.deficit_cost_sek_per_c(19.0)


def test_a_degree_at_the_end_costs_what_the_fitted_models_say_it_takes_to_win_back(
    settings, make_inputs
):
    cost = _end_cost_per_degree(settings, make_inputs, PRIOR_PUMP_PARAMS)
    assert cost > 0
    # the pump needing twice the energy per setpoint degree makes a degree twice as dear
    thirsty = replace(
        PRIOR_PUMP_PARAMS, energy_per_rts=2 * PRIOR_PUMP_PARAMS.energy_per_rts
    )
    assert _end_cost_per_degree(settings, make_inputs, thirsty) > 1.5 * cost


def test_the_objective_credits_heat_left_in_the_room_beyond_the_flat_setpoints_end(
    settings, make_inputs
):
    price = np.full(HORIZON, 1.0)
    price[8:16] = 2.5
    result = run(make_inputs(price, room_c=19.0, slab_c=28.0), settings)
    assert result.hours[-1].room_c_end > result.baseline_hours[-1].room_c_end
    hours_cost = sum(h.energy_cost_sek + h.comfort_cost_sek for h in result.hours)
    assert result.objective_sek < hours_cost  # the heat that is left does not have to be bought


def test_the_credit_for_stored_heat_stops_half_a_degree_above_the_flat_setpoints_end(
    settings, make_inputs
):
    from core.theria.plan.planner import END_CREDIT_C

    price = np.full(HORIZON, 3.0)
    price[:10] = 0.01  # nearly free heat early, expensive after
    result = run(make_inputs(price, room_c=19.5, slab_c=28.0), settings)
    ends = result.hours[-1].room_c_end - result.baseline_hours[-1].room_c_end
    assert ends <= END_CREDIT_C + 0.1


def test_when_the_flow_is_above_the_shunt_cap_a_lower_setpoint_costs_the_floor_nothing(
    settings, make_inputs
):
    # at -5 C outdoors the flow is above 40 C even at the lowest setpoint, so the floor only ever
    # receives 40 C: cutting the pump setpoint saves electricity and the room does not notice
    result = run(make_inputs(np.full(HORIZON, 1.0), t_out=-5.0), settings)
    assert all(h.rts_c == settings.rts_min_c for h in result.hours)
    for plan_hour, flat_hour in zip(result.hours, result.baseline_hours, strict=True):
        assert abs(plan_hour.room_c_end - flat_hour.room_c_end) < 0.01


def _with_midday_sun(inputs, hours=range(4, 10)):
    sun = np.zeros(HORIZON)
    sun[list(hours)] = 400.0
    return replace(inputs, sun_wm2=sun)


def run_with_sun(inputs, settings, swing):
    baseline_flow = np.minimum(
        flow_target(PRIOR_PUMP_PARAMS, inputs.t_out_c, settings.rts_base_c),
        settings.floor_mix_max_c,
    )
    room = anchored_room_params(
        PRIOR_ROOM_PARAMS,
        inputs.initial.room_c,
        inputs.t_out_c,
        baseline_flow,
        settings.lk_limit_c,
    )
    return plan(
        inputs, settings, PRIOR_PUMP_PARAMS, replace(room, sun_swing_c_per_wm2=swing)
    )


def test_the_suns_swing_warms_the_room_the_plan_is_judged_on(settings, make_inputs):
    inputs = _with_midday_sun(make_inputs(np.full(HORIZON, 1.0), room_c=18.6, slab_c=27.0))
    without = run_with_sun(inputs, settings, 0.0)
    with_sun = run_with_sun(inputs, settings, 0.006)
    sunny = slice(4, 10)
    # the level is slow, so the setpoint does not change in the sunny hours themselves, but the
    # room is warmer there: less discomfort, and a cheaper plan
    assert sum(h.comfort_cost_sek for h in with_sun.hours[sunny]) < sum(
        h.comfort_cost_sek for h in without.hours[sunny]
    )
    assert with_sun.objective_sek < without.objective_sek
    assert all(h.sun_swing_c > 0 for h in with_sun.hours[sunny])
    assert all(h.sun_swing_c == 0 for h in without.hours)


def test_a_room_the_sun_pushes_over_the_ceiling_is_not_heated_further(
    settings, make_inputs
):
    inputs = _with_midday_sun(make_inputs(np.full(HORIZON, 1.0), room_c=20.8, slab_c=30.0))
    with_sun = run_with_sun(inputs, settings, 0.01)  # level 20.8 + a swing of about 3 C
    sunny = slice(4, 10)
    assert max(h.rts_c for h in with_sun.hours[sunny]) <= settings.rts_base_c


def test_a_setpoint_range_narrower_than_the_rise_limit_still_plans(settings, make_inputs):
    narrow = replace(settings, rts_min_c=18.0, rts_max_c=19.0)
    result = run(make_inputs(daily_prices()), narrow)
    assert all(18.0 <= h.rts_c <= 19.0 for h in result.hours)


def test_the_other_rooms_cost_keeps_a_small_price_difference_from_moving_the_setpoint(
    settings, make_inputs
):
    flat_ish = np.where(
        EVENING, 1.15, 1.0
    )  # 0.15 SEK/kWh peak: not worth disturbing the rest
    result = run(make_inputs(flat_ish, room_c=20.0, slab_c=33.0), settings)
    rts = np.array(
        [h.rts_c for h in result.hours if not h.tail]
    )  # the tail is low confidence
    assert np.abs(rts - settings.rts_base_c).max() < 1.0


def test_the_plan_does_not_boost_the_flow_into_the_range_where_the_electric_heater_starts(
    settings, make_inputs
):
    cheap_night = np.where(np.arange(36) % 24 < 5, 0.3, 3.0)  # a strong reason to boost
    result = run(
        make_inputs(cheap_night, t_out=5.0, room_c=20.0, slab_c=33.0), settings
    )
    assert (
        max(h.flow_target_c for h in result.hours) <= settings.boost_flow_limit_c + 1e-9
    )


def test_in_cold_weather_the_plan_may_cut_but_never_exceed_the_flow_the_baseline_already_has(
    settings, make_inputs
):
    # at -4 C the baseline flow (about 43.5 C) is already above the boost limit
    result = run(
        make_inputs(daily_prices(), t_out=-4.0, room_c=20.0, slab_c=40.0), settings
    )
    baseline_flow = flow_target(PRIOR_PUMP_PARAMS, -4.0, settings.rts_base_c)
    assert baseline_flow > settings.boost_flow_limit_c
    assert max(h.flow_target_c for h in result.hours) <= baseline_flow + 1e-9
    assert (
        min(h.flow_target_c for h in result.hours) < baseline_flow - 1.0
    )  # it still cuts in peaks


def test_the_default_setpoint_range_is_the_old_manual_one():
    from core.theria.plan.settings import PlanSettings as Settings

    s = Settings.from_dict(BASE_SETTINGS)
    assert (s.rts_min_c, s.rts_base_c, s.rts_max_c) == (16.0, 18.0, 20.0)
    assert s.max_rts_step_per_hour_c == 2.0  # the whole range in one step
    assert s.boost_flow_limit_c == 42.0


def test_a_cooling_that_would_break_the_night_floor_makes_the_plan_push_as_hard_as_it_can(
    settings, make_inputs
):
    from dataclasses import replace

    night = replace(settings, night_start_hour=0, night_end_hour=6)
    index = np.arange(HORIZON)
    # START is 02:00 local: the night is indices 0-3. The room is expected to lose 0.6 C per
    # night hour and to win it back by day (the pattern has zero mean, like the anchoring).
    pattern = np.where(index < 4, -0.6, 2.4 / (HORIZON - 4))
    prices = np.full(HORIZON, 1.0)
    plain_inputs = make_inputs(prices, room_c=19.5)
    plain = run(plain_inputs, night)
    cold = run(replace(plain_inputs, extra_drift_c_per_h=pattern), night)
    assert cold.baseline_hours[3].room_c_end < plain.baseline_hours[3].room_c_end - 2.0
    assert (
        cold.floor_violation_hours >= 1
    ), "the night floor cannot be held in this case"
    assert sum(h.rts_c for h in cold.hours[:4]) > sum(h.rts_c for h in plain.hours[:4])


def test_a_plan_that_saves_less_than_the_threshold_stays_at_the_flat_setpoint(
    settings, make_inputs
):
    price = np.where(EVENING, 4.0, 0.8)
    inputs = make_inputs(price)
    saving = run(inputs, replace(settings, min_saving_sek=0.0))
    gain = saving.baseline_objective_sek - saving.objective_sek
    assert gain > 0

    result = run(inputs, replace(settings, min_saving_sek=gain + 1.0))

    assert {h.rts_c for h in result.hours} == {settings.rts_base_c}
    assert {h.reason["binding"] for h in result.hours} == {"below_min_saving"}
    assert result.objective_sek == result.baseline_objective_sek


def test_a_plan_that_saves_more_than_the_threshold_is_kept(settings, make_inputs):
    price = np.where(EVENING, 4.0, 0.8)
    inputs = make_inputs(price)
    gain = run(inputs, replace(settings, min_saving_sek=0.0))
    gain = gain.baseline_objective_sek - gain.objective_sek

    result = run(inputs, replace(settings, min_saving_sek=gain - 0.01))

    assert len({h.rts_c for h in result.hours}) > 1
