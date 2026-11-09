from dataclasses import replace
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core.theria.plan.heat_shift import (
    find_trades,
    make_plan,
    shift_window_hours,
)
from core.theria.plan.planner import PlanInputs
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS, RoomState
from core.theria.plan.shift_rules import SalonStatus, ShiftReadings, ShiftWindow

MONDAY = [
    2.86, 2.77, 2.75, 2.78, 2.72, 2.91, 3.72, 4.59, 4.88, 4.69, 4.41, 4.21,
    4.23, 4.19, 4.63, 6.32, 6.22, 7.44, 6.54, 5.56, 4.56, 3.73, 3.48, 2.92,
]  # fmt: skip
STOCKHOLM = ZoneInfo("Europe/Stockholm")
MIDNIGHT = datetime(2026, 11, 9, 0, 0, tzinfo=STOCKHOLM).astimezone(UTC)


def shifting(settings):
    return replace(
        settings,
        plan_strategy="heat_shift",
        rts_min_c=15.0,
        rts_max_c=22.0,
        rts_base_c=18.0,
    )


def inputs_for(prices, past=(), t_out=7.5, start_hour=0, shift=None):
    n = len(prices)
    return PlanInputs(
        hour_starts=[MIDNIGHT + timedelta(hours=start_hour + i) for i in range(n)],
        price_sek_per_kwh=np.array(prices, dtype=float),
        extreme=np.zeros(n, bool),
        t_out_c=np.full(n, t_out),
        sun_wm2=np.zeros(n),
        initial=RoomState(19.4, 30.0, 0.0),
        previous_rts_c=18.0,
        past_prices_sek_per_kwh=np.array(past, dtype=float),
        shift=shift,
    )


def planned(settings, inputs):
    result = make_plan(inputs, shifting(settings), PRIOR_PUMP_PARAMS, PRIOR_ROOM_PARAMS)
    return result, [h.rts_c for h in result.hours]


@pytest.mark.parametrize("t_out", [7.5, 0.0, -5.0, 12.0])
def test_a_colder_day_allows_a_shorter_cut(t_out):
    colder = shift_window_hours(t_out - 5.0, 19.4, 0.5)
    assert 1 <= colder <= shift_window_hours(t_out, 19.4, 0.5) <= 8


def test_the_window_follows_the_measured_cooling():
    assert shift_window_hours(7.5, 19.4, 0.5) == 5
    assert shift_window_hours(0.0, 19.4, 0.5) == 3


def test_a_smaller_allowed_swing_gives_a_shorter_window():
    assert shift_window_hours(7.5, 19.4, 0.1) == 1


def test_the_dearest_period_is_sold_and_the_cheapest_before_it_bought():
    assert find_trades(MONDAY, 5, 0.5) == [(0, 15)]


def test_a_small_price_difference_is_not_worth_a_trade():
    assert find_trades([3.0, 3.1, 3.0, 3.1, 3.0, 3.1, 3.0, 3.1], 2, 0.5) == []


def test_two_price_peaks_with_a_cheap_valley_between_give_two_trades():
    prices = [1.0, 1.0, 5.0, 5.0, 1.0, 1.0, 6.0, 6.0, 2.0, 2.0]
    assert find_trades(prices, 2, 0.5) == [(0, 2), (4, 6)]


def test_monday_lowers_in_the_dearest_hours_and_only_there(settings):
    _, rts = planned(settings, inputs_for(MONDAY))
    assert rts[15:20] == [17.0] * 5
    assert all(r >= 18.0 for i, r in enumerate(rts) if i not in range(15, 20))


def test_monday_raises_only_in_the_cheapest_hours_before_the_cut(settings):
    _, rts = planned(settings, inputs_for(MONDAY))
    assert rts[:5] == [19.0] * 5
    assert all(r <= 18.0 for r in rts[5:])


def test_the_plan_never_goes_below_one_shift_from_the_base(settings):
    _, rts = planned(settings, inputs_for(MONDAY))
    assert min(rts) == 17.0


def test_the_floor_loops_close_in_the_sell_hours(settings):
    result, _ = planned(replace(settings, plan_closes_valves=True), inputs_for(MONDAY))
    sells = [h for h in result.hours if h.reason["binding"] == "shift_sell"]
    assert sells and all(h.valves_closed and h.thermostat == "closing" for h in sells)
    others = [h for h in result.hours if h.reason["binding"] != "shift_sell"]
    assert not any(h.valves_closed for h in others)


def test_a_replan_inside_the_cut_keeps_the_cut(settings):
    at_four_pm = 16
    _, rts = planned(
        settings,
        inputs_for(
            MONDAY[at_four_pm:], past=MONDAY[:at_four_pm], start_hour=at_four_pm
        ),
    )
    assert rts[:4] == [17.0] * 4


def test_flat_prices_stay_at_the_base(settings):
    result, rts = planned(settings, inputs_for([3.0] * 24))
    assert set(rts) == {18.0}
    assert result.hours[0].reason["binding"] == "shift_none"


def test_the_plan_says_which_hours_buy_and_which_sell(settings):
    result, _ = planned(settings, inputs_for(MONDAY))
    assert result.hours[1].reason["binding"] == "shift_buy"
    assert result.hours[16].reason["binding"] == "shift_sell"
    assert result.hours[10].reason["binding"] == "shift_none"


def test_the_saving_comes_from_the_price_difference(settings):
    result, _ = planned(settings, inputs_for(MONDAY))
    plan_cost = sum(h.energy_cost_sek for h in result.hours)
    flat_cost = sum(h.energy_cost_sek for h in result.baseline_hours)
    assert 0 < flat_cost - plan_cost < 20


def test_the_room_and_salon_effect_of_the_cut_is_modelled(settings):
    result, _ = planned(settings, inputs_for(MONDAY))
    by_hour = dict(enumerate(result.hours))
    drop_main = by_hour[19].room_c_end - result.baseline_hours[19].room_c_end
    assert -0.5 < drop_main < 0.0
    assert by_hour[3].salon_delta_c > 0.0
    assert -0.5 < by_hour[19].salon_delta_c < by_hour[3].salon_delta_c


def at_three_pm(shift):
    return inputs_for(MONDAY[15:], past=MONDAY[:15], start_hour=15, shift=shift)


def hour(n):
    return MIDNIGHT + timedelta(hours=n)


def test_a_window_is_committed_when_it_starts(settings):
    result, _ = planned(settings, at_three_pm(ShiftReadings(19.5)))
    assert result.shift_window == ShiftWindow(hour(20), -1.0, None)


def test_no_window_is_committed_between_windows(settings):
    prices = MONDAY[10:]
    result, _ = planned(
        settings,
        inputs_for(prices, past=MONDAY[:10], start_hour=10, shift=ShiftReadings(19.5)),
    )
    assert result.shift_window is None


def test_a_started_window_runs_to_its_end_whatever_the_prices_say(settings):
    window = ShiftWindow(hour(18), -1.0)
    _, rts = planned(
        settings,
        inputs_for(
            [3.0] * 8,
            past=[3.0] * 15,
            start_hour=15,
            shift=ShiftReadings(19.5, None, window),
        ),
    )
    assert rts == [17.0] * 3 + [18.0] * 5


def test_a_finished_window_no_longer_holds(settings):
    window = ShiftWindow(hour(15), -1.0)
    _, rts = planned(
        settings,
        inputs_for(
            [3.0] * 4,
            past=[3.0] * 15,
            start_hour=15,
            shift=ShiftReadings(19.5, None, window),
        ),
    )
    assert set(rts) == {18.0}


def test_the_thermostat_is_the_guard_of_a_cold_room_so_the_sell_goes_on(settings):
    result, rts = planned(settings, at_three_pm(ShiftReadings(17.9)))
    assert rts[:5] == [17.0] * 5


def test_a_salon_guard_returns_the_pump_but_the_thermostats_keep_closing(settings):
    result, rts = planned(
        replace(settings, plan_closes_valves=True),
        at_three_pm(ShiftReadings(19.5, SalonStatus(18.0, 19.0))),
    )
    assert rts[:5] == [18.0] * 5
    assert result.hours[0].reason["guard"] == "salon_floor"
    assert all(h.thermostat == "closing" for h in result.hours[:5])


def test_a_sell_ended_by_a_guard_stays_ended_when_the_room_recovers(settings):
    window = ShiftWindow(hour(20), -1.0, "salon_floor")
    result, rts = planned(
        settings,
        inputs_for(
            MONDAY[16:],
            past=MONDAY[:16],
            start_hour=16,
            shift=ShiftReadings(19.5, None, window),
        ),
    )
    assert rts[:4] == [18.0] * 4
    assert result.shift_window == window


def test_a_salon_under_its_floor_ends_the_sell(settings):
    salon = SalonStatus(18.9, 19.0)
    result, rts = planned(settings, at_three_pm(ShiftReadings(19.5, salon)))
    assert rts[:5] == [18.0] * 5
    assert result.hours[0].reason["guard"] == "salon_floor"


def test_without_readings_nothing_is_committed(settings):
    result, _ = planned(settings, inputs_for(MONDAY))
    assert result.shift_window is None


def test_a_cut_is_paid_for_by_as_much_buying_before_it(settings):
    prices = [1.0, 1.0, 1.0, 8.0, 9.0, 9.0, 9.0, 9.0, 1.0, 1.0, 1.0, 1.0]
    _, rts = planned(settings, inputs_for(prices))
    net = sum(r - 18.0 for r in rts)
    moved = sum(abs(r - 18.0) for r in rts)
    assert moved > 0 and abs(net) <= 0.1 * moved


def closed_energy(settings, closes):
    result, _ = planned(replace(settings, plan_closes_valves=closes), inputs_for(MONDAY))
    return next(
        h.energy_kwh for h in result.hours if h.reason["binding"] == "shift_sell"
    )


def test_a_closed_sell_hour_still_uses_the_measured_share_of_the_energy(settings):
    assert closed_energy(settings, True) == pytest.approx(
        0.6 * closed_energy(settings, False)
    )
