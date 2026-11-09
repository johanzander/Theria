import math
from datetime import UTC, datetime, time, timedelta

import pytest

from core.theria.radiators.salon_plan import SalonModel, plan_salon
from core.theria.radiators.settings import DaySchedule, RadiatorPrefs

TZ = "Europe/Stockholm"
MODEL = SalonModel(c_j_per_k=2.33e6, u_w_per_k=51.7, heat_power_w=1200.0)
PREFS = RadiatorPrefs()
BOUND = PREFS.salon_open_target_c + PREFS.salon_margin_c  # 20.3
# Monday 2026-11-02 06:00 local (UTC+1) is 05:00 UTC
MONDAY_06 = datetime(2026, 11, 2, 5, tzinfo=UTC)


def run(
    prices,
    start=MONDAY_06,
    start_mean=19.5,
    t_out=5.0,
    bias=0.5,
    prefs=PREFS,
    min_saving_sek=0.2,
    model=MODEL,
):
    hours = [start + timedelta(hours=k) for k in range(len(prices))]
    flags = [prefs.open_in_hour(h, TZ) for h in hours]
    return plan_salon(
        hours,
        prices,
        [t_out] * len(hours),
        flags,
        start_mean,
        prefs,
        model,
        bias,
        min_saving_sek,
    )


def test_flat_prices_hold_the_target_without_movement():
    plan = run([1.0] * 24)
    assert all("flat" in h.reason for h in plan)
    open_setpoints = {h.setpoint_c for h in plan if h.open}
    assert len(open_setpoints) == 1


def test_a_dear_hour_inside_opening_hours_is_coasted_after_a_pre_heat():
    prices = [0.5] * 24
    prices[8] = 3.0  # 14:00-15:00 local
    plan = run(prices, t_out=12.0)
    assert plan[7].heat and plan[7].mean_end_c >= PREFS.salon_preheat_c - 0.11
    assert not plan[8].heat
    assert plan[8].mean_end_c >= BOUND - 1e-6
    assert "flat" not in plan[8].reason


def test_the_mean_never_falls_below_target_plus_margin_while_open():
    prices = [0.4, 2.5, 0.6, 3.0, 0.5, 2.8, 0.7, 3.2, 0.5, 2.9, 0.6, 3.1] * 2
    for h in run(prices, t_out=10.0):
        if h.open:
            assert h.mean_end_c >= BOUND - 1e-6


def evening_to_noon(night_price=3.0):
    # 19:00 local Monday to 11:00 Tuesday: closed until 09:00
    start = datetime(2026, 11, 2, 18, tzinfo=UTC)
    prices = [night_price if k < 10 else 0.5 for k in range(17)]
    return run(prices, start=start, start_mean=19.5)


def test_closed_hours_may_cool_down_to_but_not_below_the_closed_floor():
    closed = [h for h in evening_to_noon() if not h.open]
    assert min(h.mean_end_c for h in closed) >= PREFS.salon_closed_floor_c - 1e-6
    assert min(h.mean_end_c for h in closed) < 18.5


def test_the_room_is_back_at_target_when_opening():
    plan = evening_to_noon()
    first_open = next(i for i, h in enumerate(plan) if h.open)
    assert plan[first_open - 1].mean_end_c >= BOUND - 1e-6


def test_a_missing_price_hour_is_planned_neutrally_and_named():
    prices = [0.5] * 12
    prices[5] = math.nan
    plan = run(prices)
    assert "no price" in plan[5].reason
    assert math.isnan(plan[5].price_sek_per_kwh)
    assert all(h.mean_end_c >= BOUND - 1e-6 for h in plan if h.open)


def test_a_short_horizon_without_tomorrows_prices_is_valid():
    plan = run([0.5, 0.6, 0.4, 3.0, 0.5, 0.5, 0.6, 0.5, 0.4, 0.5, 0.5, 0.6])
    assert len(plan) == 12
    assert all(h.mean_end_c >= BOUND - 1e-6 for h in plan if h.open)


def test_the_dst_day_of_25_hours_is_planned_per_absolute_hour():
    start = datetime(2026, 10, 24, 22, tzinfo=UTC)  # Sunday 2026-10-25 00:00 local
    prices = [0.5 if k % 5 else 2.5 for k in range(25)]
    plan = run(prices, start=start, start_mean=19.5)
    assert len(plan) == 25
    assert sum(h.open for h in plan) == 6  # Sunday 10:00-16:00
    assert all(h.mean_end_c >= BOUND - 1e-6 for h in plan if h.open)


def test_setpoint_is_a_half_degree_step_follows_the_bias_and_is_capped():
    prices = [0.5] * 24
    prices[8] = 3.0
    plan = run(prices, t_out=12.0, bias=2.0)
    for h in plan:
        assert h.setpoint_c * 2 == int(h.setpoint_c * 2)
        assert h.setpoint_c <= PREFS.salon_max_setpoint_c
    assert plan[7].setpoint_c == PREFS.salon_max_setpoint_c  # 20.0 + 2.0
    assert [h.setpoint_c for h in run(prices, t_out=12.0, bias=5.0)] == [
        h.setpoint_c for h in plan
    ]


def test_a_tiny_saving_is_not_taken():
    prices = [0.5 if k % 2 else 0.5001 for k in range(24)]
    assert all("flat" in h.reason for h in run(prices))


def test_infeasible_requirements_hold_and_say_so():
    start = datetime(2026, 11, 2, 7, tzinfo=UTC)  # 08:00 local, open at 09:00
    plan = run([0.5], start=start, start_mean=12.0)
    assert len(plan) == 1
    assert "infeasible" in plan[0].reason
    assert plan[0].heat
    assert plan[0].setpoint_c == PREFS.salon_max_setpoint_c


def test_a_cold_day_is_planned_hour_by_hour_not_given_up():
    strong = SalonModel(c_j_per_k=2.33e6, u_w_per_k=51.7, heat_power_w=1800.0)
    plan = run([0.5] * 24, t_out=-5.0, start_mean=19.5, model=strong)
    assert not any("infeasible" in h.reason for h in plan)
    assert all(h.mean_end_c >= BOUND - 1e-6 for h in plan if h.open)


def test_only_the_hour_that_cannot_be_met_runs_at_full_power():
    start = datetime(2026, 11, 2, 0, tzinfo=UTC)  # 01:00 local, closed
    plan = run([0.5] * 8, start=start, start_mean=15.5)
    assert (
        "infeasible" in plan[0].reason
        and plan[0].setpoint_c == PREFS.salon_max_setpoint_c
    )
    assert not any("infeasible" in h.reason for h in plan[2:])
    assert plan[-1].setpoint_c < PREFS.salon_max_setpoint_c


@pytest.mark.parametrize("start_mean", [20.8, 23.0])
def test_the_pre_heat_cap_holds_when_the_salon_starts_above_it(start_mean):
    prices = [0.5] * 4 + [3.0] * 3 + [0.5] * 10
    plan = run(prices, start_mean=start_mean, t_out=12.0)
    previous = start_mean
    for h in plan:
        assert h.mean_end_c <= max(PREFS.salon_preheat_c, previous) + 1e-6
        previous = h.mean_end_c


def test_an_opening_at_half_past_the_hour_finds_the_shop_warm():
    schedule = {**PREFS.salon_schedule, "mon": DaySchedule(time(9, 30), time(19, 0))}
    prefs = RadiatorPrefs(salon_schedule=schedule)
    plan = run([0.5] * 8, prefs=prefs, start_mean=18.5)  # 06:00 local onwards
    assert plan[3].open and not plan[2].open  # 09:00-10:00 holds the opening at 09:30
    assert plan[2].mean_end_c >= BOUND - 1e-6
