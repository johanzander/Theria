import numpy as np
import pytest

from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, flow_target
from core.theria.plan.room_model import (
    PRIOR_ROOM_PARAMS,
    RoomParams,
    RoomState,
    anchored_room_params,
    simulate,
    step,
    valve_open,
)


def test_valves_close_as_room_reaches_the_lk_setpoint():
    assert valve_open(19.0, 21.0) == 1.0
    assert valve_open(20.75, 21.0) == pytest.approx(0.5)
    assert valve_open(21.5, 21.0) == 0.0


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


def test_anchored_drift_keeps_the_baseline_near_the_current_room_temperature():
    hours = np.arange(72)
    t_out = 3 + 4 * np.sin(2 * np.pi * hours / 24)
    baseline_flow = flow_target(PRIOR_PUMP_PARAMS, t_out, 18.0)
    anchored = anchored_room_params(PRIOR_ROOM_PARAMS, 20.0, t_out, baseline_flow, 21.0)
    room = simulate(
        anchored, RoomState(20.0, 20.0, 0.0), baseline_flow, t_out, np.zeros(72), 21.0
    )
    assert np.abs(room - 20.0).max() < 0.4


def test_anchored_model_still_responds_to_a_different_flow_temperature():
    t_out = np.full(72, 3.0)
    baseline_flow = flow_target(PRIOR_PUMP_PARAMS, t_out, 18.0)
    lk = 30.0  # valves stay open: only the dynamics are tested here
    anchored = anchored_room_params(PRIOR_ROOM_PARAMS, 20.0, t_out, baseline_flow, lk)
    start = RoomState(20.0, 20.0, 0.0)
    base = simulate(anchored, start, baseline_flow, t_out, np.zeros(72), lk)
    warm = simulate(anchored, start, baseline_flow + 5.0, t_out, np.zeros(72), lk)
    # 5 C more flow for three days: gain * 5 * (1 - exp(-rate * 3)) = about 0.9 C
    assert warm[-1] - base[-1] == pytest.approx(
        0.39 * 5 * (1 - np.exp(-0.21 * 3)), abs=0.1
    )
