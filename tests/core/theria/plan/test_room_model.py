import pytest

from core.theria.plan.room_model import (
    PRIOR_ROOM_PARAMS,
    RoomParams,
    step,
    valve_open,
)


def test_valves_close_as_room_reaches_the_lk_setpoint():
    # measured 29 Oct - 10 Nov 2026: the loops start to open 0.75 C above the target, all open 0.95 C below it
    assert valve_open(20.0, 21.0) == 1.0
    assert valve_open(21.0, 21.0) == pytest.approx(0.75 / 1.7)
    assert valve_open(21.5, 21.0) == pytest.approx(0.25 / 1.7)
    assert valve_open(21.8, 21.0) == 0.0


def test_hot_slab_warms_room_and_closed_valves_stop_it():
    params = RoomParams(6.0, 0.04, 0.05, 0.0, 3.0, 0.0)
    open_room, _, _ = step(params, 19.0, 30.0, 0.0, 35.0, 5.0, 0.0, 21.0)
    closed_room, closed_slab, _ = step(params, 19.0, 30.0, 0.0, 35.0, 5.0, 0.0, 18.0)
    assert open_room > 19.0 - params.k_out * (19.0 - 5.0)
    assert closed_room < 19.0  # only loss to outdoors
    assert closed_slab < 30.0  # slab relaxes towards the room


def test_sun_warms_the_room_with_delay():
    params = RoomParams(6.0, 0.04, 0.05, 0.0004, 3.0, 0.0)
    first = step(params, 19.0, 19.0, 0.0, 19.0, 19.0, 600.0, 19.0)
    assert first[0] - 19.0 < params.k_sun * 600.0  # not instant
    assert first[2] > 0.0


def test_hot_water_scale_closes_the_valves():
    params = RoomParams(6.0, 0.04, 0.05, 0.0, 3.0, 0.0)
    with_water = step(params, 19.0, 30.0, 0.0, 40.0, 5.0, 0.0, 21.0)
    without = step(params, 19.0, 30.0, 0.0, 40.0, 5.0, 0.0, 21.0, valve_scale=0.0)
    assert with_water[0] > without[0]


def test_prior_is_the_measured_response_of_the_house():
    p = PRIOR_ROOM_PARAMS
    gain = p.k_slab / (p.k_slab + p.k_out)
    time_constant_days = 1 / ((p.k_slab + p.k_out) * 24)
    assert gain == pytest.approx(0.39)
    assert time_constant_days == pytest.approx(1 / 0.21)
