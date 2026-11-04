"""Ground-floor room model used by the planner.

Hourly dynamics: floor heat couples to the room through a slab state that follows the flow
temperature while the LK valves are open, the room loses heat to the outdoors, and the sun
adds heat. The response is the measured one (`with_measured_response`). What else the room does
(stove, people, weather swings) is outside the model: the plan adds the measured heat by day
(`day_heat_c_per_h`) and the disturbance, and every plan starts from the measured room.
"""

from dataclasses import dataclass, replace

import numpy as np

Array = float | np.ndarray  # scalars and numpy arrays are both accepted

VALVE_RAMP_C = 0.5  # valves close over the last 0.5 C below the LK setpoint
# Measured on real plan nights with the loops open and closed, 27 Oct - 4 Nov 2026, outdoor 4-13 C
# (documentation/ivt-heat-pump.md, "Closed loops measured"). The daily fit's response is about half as fast.
MEASURED_K_SLAB = 0.0065
MEASURED_K_OUT = 0.0113
# What the house gains by day on top of that (sun, people, the stove as it was used those days),
# per local hour block; nothing at night. A mean, so a day without a fire comes in cooler: the next
# plan starts from the measured room.
DAY_HEAT_C_PER_H = ((6, 0.0), (12, 0.093), (18, 0.126), (24, 0.028))


@dataclass(frozen=True)
class RoomParams:
    """Parameters of the room model (per hour).

    d_room = valve * k_slab * (slab - room) - k_out * (room - t_out)
             + k_sun * sun_filtered + drift
    slab relaxes towards the flow temperature while valves are open and towards
    the room temperature while they are closed, with time constant tau_slab_h.
    sun_filtered is the irradiance smoothed with time constant sun_lag_h.
    """

    tau_slab_h: float
    k_slab: float
    k_out: float
    k_sun: float
    sun_lag_h: float
    drift_c_per_h: float
    # The sun lifts the room above its slow level by this much per W/m2 of filtered irradiation:
    # a swing on top of the level, not a heating rate (the room is slow, the sun is quick).
    sun_swing_c_per_wm2: float = 0.0


# Medians of the four winters' daily fits on Johan's data (2026-10-11): one degree of flow
# temperature moves the room about 0.39 C in the long run, and the room closes about 21 % of
# the gap per day (time constant about 4.7 days). The sun and the drift are not modelled.
PRIOR_GAIN_C_PER_C = 0.39
PRIOR_RATE_PER_DAY = 0.21
PRIOR_ROOM_PARAMS = RoomParams(
    tau_slab_h=1.0,
    k_slab=PRIOR_GAIN_C_PER_C * PRIOR_RATE_PER_DAY / 24,
    k_out=(1 - PRIOR_GAIN_C_PER_C) * PRIOR_RATE_PER_DAY / 24,
    k_sun=0.0,
    sun_lag_h=1.0,
    drift_c_per_h=0.0,
)


@dataclass(frozen=True)
class RoomState:
    room_c: float
    slab_c: float
    sun_filtered_wm2: float


def with_measured_response(params: "RoomParams") -> "RoomParams":
    """The room model with the measured response, and no drift: what the house adds comes from
    `day_heat_c_per_h` and from the disturbance, never from the room's temperature right now."""
    return replace(
        params, k_slab=MEASURED_K_SLAB, k_out=MEASURED_K_OUT, drift_c_per_h=0.0
    )


def day_heat_c_per_h(local_hour: int) -> float:
    """The measured extra heat in this local hour."""
    return next(heat for end, heat in DAY_HEAT_C_PER_H if local_hour < end)


def valve_open(room_c: Array, lk_setpoint_c: Array) -> Array:
    """Fraction (0-1) of the loops open; scalars and numpy arrays."""
    return np.clip((lk_setpoint_c - room_c) / VALVE_RAMP_C, 0.0, 1.0)


def step(
    params: RoomParams,
    room_c: Array,
    slab_c: Array,
    sun_filtered_wm2: Array,
    flow_c: Array,
    t_out_c: Array,
    sun_wm2: Array,
    lk_setpoint_c: float,
    dt_h: float = 1.0,
    valve_scale: Array = 1.0,
) -> tuple[Array, Array, Array]:
    """Advance one step. Works on scalars and numpy arrays.

    Returns (room_c, slab_c, sun_filtered_wm2) at the end of the step.
    """
    valve = valve_open(room_c, lk_setpoint_c) * valve_scale
    d_room = (
        valve * params.k_slab * (slab_c - room_c)
        - params.k_out * (room_c - t_out_c)
        + params.k_sun * sun_filtered_wm2
        + params.drift_c_per_h
    ) * dt_h
    slab_target = valve * flow_c + (1.0 - valve) * room_c
    slab_next = slab_c + (slab_target - slab_c) * (
        1.0 - np.exp(-dt_h / params.tau_slab_h)
    )
    sun_next = sun_filtered_wm2 + (sun_wm2 - sun_filtered_wm2) * (
        1.0 - np.exp(-dt_h / params.sun_lag_h)
    )
    return room_c + d_room, slab_next, sun_next


def simulate(
    params: RoomParams,
    initial: RoomState,
    flow_c: np.ndarray,
    t_out_c: np.ndarray,
    sun_wm2: np.ndarray,
    lk_setpoint_c: float,
) -> np.ndarray:
    """Room temperature at the end of each hour for the given hourly forcing."""
    room: Array = initial.room_c
    slab: Array = initial.slab_c
    sun_f: Array = initial.sun_filtered_wm2
    out = np.empty(len(flow_c))
    for i in range(len(flow_c)):
        room, slab, sun_f = step(
            params, room, slab, sun_f, flow_c[i], t_out_c[i], sun_wm2[i], lk_setpoint_c
        )
        out[i] = room
    return out
