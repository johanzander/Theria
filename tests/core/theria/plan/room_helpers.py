"""Planner tests need a room that holds its level on a flat setpoint, so that only the plan moves it."""

from dataclasses import replace

import numpy as np

from core.theria.plan.room_model import RoomParams, valve_open


def anchored_room_params(
    params: RoomParams,
    room_c: float,
    t_out_c: np.ndarray,
    flow_c: np.ndarray,
    lk_setpoint_c: float,
) -> RoomParams:
    valve = float(valve_open(room_c, lk_setpoint_c))
    drift = -(
        valve * params.k_slab * (float(np.mean(flow_c)) - room_c)
        + params.k_out * (float(np.mean(t_out_c)) - room_c)
    )
    return replace(params, drift_c_per_h=drift)
