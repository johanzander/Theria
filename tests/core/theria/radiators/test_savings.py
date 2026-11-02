from datetime import UTC, datetime, timedelta

import pytest

from core.theria.radiators.guest_plan import GuestHour
from core.theria.radiators.salon_plan import SalonHour, SalonModel
from core.theria.radiators.savings import (
    guest_saving_sek,
    salon_saving_sek,
    would_lift_block,
)

MODEL = SalonModel(c_j_per_k=2.33e6, u_w_per_k=51.7, heat_power_w=1200.0)
START = datetime(2026, 11, 2, 5, tzinfo=UTC)
T_OUT = 10.0


def salon(levels, prices):
    return [
        SalonHour(START + timedelta(hours=k), True, True, level, 20.0, price, "x")
        for k, (level, price) in enumerate(zip(levels, prices, strict=True))
    ]


def test_a_plan_equal_to_the_baseline_saves_nothing():
    plan = salon([20.0] * 6, [1.0] * 6)
    assert salon_saving_sek(plan, 20.0, [T_OUT] * 6, MODEL) == pytest.approx(0.0)


def test_moving_heat_from_dear_to_cheap_hours_saves_money():
    prices = [0.5, 0.5, 3.0, 3.0, 0.5, 0.5]
    # cool through the dear hours, then recover the heat in the cheap ones
    plan = salon([20.5, 20.5, 19.5, 18.7, 20.0, 20.0], prices)
    assert salon_saving_sek(plan, 20.0, [T_OUT] * 6, MODEL) > 0


def test_a_plan_that_heats_more_than_the_baseline_costs_money():
    plan = salon([21.0] * 4, [1.0] * 4)
    assert salon_saving_sek(plan, 20.0, [T_OUT] * 4, MODEL) < 0


def guest(setpoints, prices):
    return [
        GuestHour(START + timedelta(hours=k), s, "x") for k, s in enumerate(setpoints)
    ]


def test_guest_saving_is_labelled_indicative_and_zero_for_equal_setpoints():
    prices = [1.0, 2.0]
    figure, label = guest_saving_sek(
        {"climate.a": guest([20.0, 20.0], prices)}, {"climate.a": 20.0}, prices, 100.0
    )
    assert figure == pytest.approx(0.0)
    assert label == "indicative"


def test_guest_cut_in_a_dear_hour_saves_and_is_signed():
    prices = [1.0, 2.0]
    plans = {"climate.a": guest([21.0, 19.0], prices)}
    figure, _ = guest_saving_sek(plans, {"climate.a": 20.0}, prices, 100.0)
    # -1 K at 1 SEK and +1 K at 2 SEK on 100 W/K in one room: 0.1 kWh each way
    assert figure == pytest.approx(0.1 * 2.0 - 0.1 * 1.0)


def test_the_block_is_lifted_only_when_all_callers_are_controlled():
    calling = {"climate.vantsidan": True, "climate.bedroom": True}
    assert would_lift_block(calling, {"climate.vantsidan", "climate.bedroom"}, True)
    assert not would_lift_block(calling, {"climate.vantsidan"}, True)


def test_the_block_is_not_lifted_when_a_room_is_below_its_bound():
    calling = {"climate.vantsidan": True}
    assert not would_lift_block(calling, {"climate.vantsidan"}, False)


def test_nothing_calling_means_nothing_to_lift():
    assert not would_lift_block(
        {"climate.vantsidan": False}, {"climate.vantsidan"}, True
    )
