"""What an hour of a plan holds, and the flat-setpoint baseline every plan is compared with.

The heat shift (`heat_shift.py`) decides the setpoint per hour; this module works out what an hour at a
setpoint means for the floor, the room and the electricity, and what the flat base setpoint would give.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt

from core.theria.plan.pump_model import (
    PumpParams,
    aux_kwh_per_hour,
    flow_target,
    total_kwh_per_hour,
)
from core.theria.plan.room_model import RoomParams, RoomState, step
from core.theria.plan.settings import PlanSettings
from core.theria.plan.shift_rules import ShiftReadings, ShiftWindow

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PlanInputs:
    """Hourly forcing for the horizon. Arrays have one entry per hour."""

    hour_starts: list[datetime]
    price_sek_per_kwh: FloatArray
    t_out_c: FloatArray
    sun_wm2: FloatArray
    initial: RoomState
    previous_rts_c: float
    # What the room does on its own, per hour (the measured heat by day); None when nothing is known.
    extra_drift_c_per_h: FloatArray | None = None
    # Prices from local midnight up to the first hour: the heat shift finds its windows from there.
    past_prices_sek_per_kwh: FloatArray | None = None
    # What the heat shift's guards read, and the window that has started; None: no guards
    shift: ShiftReadings | None = None
    # The heating electricity per hour (rows) and setpoint (columns) from the energy forecast's model;
    # None until it has been fitted, and the pump model is used
    energy_setpoints_c: FloatArray | None = None
    energy_kwh_table: FloatArray | None = None


@dataclass(frozen=True)
class HourPlan:
    start: datetime
    rts_c: float
    price_sek_per_kwh: float
    floor_c: float
    flow_target_c: float
    energy_kwh: float
    room_c_end: float
    slab_c_end: float
    energy_cost_sek: float
    reason: dict[str, Any]
    # What the sun adds on top of room_c_end (the slow level) at the end of the hour; the room the
    # sensors show is the level plus this.
    sun_swing_c: float = 0.0
    # The floor loops are closed this hour: the room only loses heat and the pump runs for what is
    # left (shift_closed_energy_share of its energy).
    valves_closed: bool = False
    # The heat shift's thermostats this hour: closing, opening or normal.
    thermostat: str = "normal"
    t_out_c: float = float("nan")  # the outdoor temperature the hour was planned with
    # What the setpoint's change does to each zone against the flat setpoint (house_model), in C.
    main_delta_c: float = 0.0
    salon_delta_c: float = 0.0


@dataclass(frozen=True)
class Plan:
    hours: list[HourPlan]
    baseline_hours: list[HourPlan]  # same model, flat rts_base
    objective_sek: float  # the plan's electricity cost over the horizon
    baseline_objective_sek: float
    floor_violation_hours: int
    shift_window: ShiftWindow | None = None  # the heat shift's window that has started


def sun_filtered_sequence(
    room: RoomParams, initial: float, sun: FloatArray
) -> FloatArray:
    """Filtered sun at the start of each hour."""
    out = np.empty(len(sun))
    value = initial
    k = 1.0 - np.exp(-1.0 / room.sun_lag_h)
    for i, s in enumerate(sun):
        out[i] = value
        value = value + (s - value) * k
    return out


class HourModel:
    """The floor, the room and the electricity of each hour of one horizon."""

    def __init__(
        self,
        inputs: PlanInputs,
        settings: PlanSettings,
        pump: PumpParams,
        room: RoomParams,
    ):
        self.inputs = inputs
        self.settings = settings
        self.pump = pump
        self.room = room
        self.horizon = len(inputs.hour_starts)
        self.extra_drift = (
            np.zeros(self.horizon)
            if inputs.extra_drift_c_per_h is None
            else np.asarray(inputs.extra_drift_c_per_h, dtype=float)
        )
        self.sun_f = sun_filtered_sequence(
            room, inputs.initial.sun_filtered_wm2, inputs.sun_wm2
        )
        sun_end = self.sun_f + (inputs.sun_wm2 - self.sun_f) * (
            1.0 - np.exp(-1.0 / room.sun_lag_h)
        )
        self.swing = room.sun_swing_c_per_wm2 * sun_end
        # the room is judged at the end of each hour, so the night ends when the state at 06:00 is reached
        night = np.array(
            [
                settings.in_night(start + timedelta(hours=1))
                for start in inputs.hour_starts
            ],
            dtype=bool,
        )
        self.floors = np.where(night, settings.night_floor_c, settings.floor_c)

    def floor_flow(self, t: int, rts_c: float) -> float:
        """The flow temperature the floor sees: what the pump sends, up to the shunts' cap."""
        pump_flow = float(flow_target(self.pump, self.inputs.t_out_c[t], rts_c))
        return min(pump_flow, self.settings.floor_mix_max_c)

    def energy_kwh(self, t: int, rts_c: float, closed: bool = False) -> float:
        table = self.inputs.energy_kwh_table
        setpoints = self.inputs.energy_setpoints_c
        if table is None or setpoints is None:
            kwh = float(total_kwh_per_hour(self.pump, self.inputs.t_out_c[t], rts_c))
        else:
            kwh = float(np.interp(rts_c, setpoints, table[t])) + float(
                aux_kwh_per_hour(self.pump, self.inputs.t_out_c[t], rts_c)
            )
        return kwh * self.settings.shift_closed_energy_share if closed else kwh

    def hour_plan(
        self,
        t: int,
        rts_c: float,
        room_end: float,
        slab_end: float,
        reason: dict[str, Any],
        closed: bool = False,
    ) -> HourPlan:
        energy = self.energy_kwh(t, rts_c, closed)
        price = float(self.inputs.price_sek_per_kwh[t])
        return HourPlan(
            start=self.inputs.hour_starts[t],
            rts_c=rts_c,
            price_sek_per_kwh=price,
            floor_c=float(self.floors[t]),
            flow_target_c=float(flow_target(self.pump, self.inputs.t_out_c[t], rts_c)),
            energy_kwh=energy,
            room_c_end=room_end,
            slab_c_end=slab_end,
            energy_cost_sek=price * energy,
            reason=reason,
            sun_swing_c=float(self.swing[t]),
            valves_closed=closed,
            t_out_c=float(self.inputs.t_out_c[t]),
        )

    def flat_plan(self, rts_c: float) -> list[HourPlan]:
        """The same models with a constant room setpoint: the comparison baseline."""
        room_c, slab_c = self.inputs.initial.room_c, self.inputs.initial.slab_c
        hours: list[HourPlan] = []
        for t in range(self.horizon):
            flow = self.floor_flow(t, rts_c)
            room_next, slab_next, _ = step(
                self.room,
                room_c,
                slab_c,
                self.sun_f[t],
                flow,
                self.inputs.t_out_c[t],
                self.inputs.sun_wm2[t],
                self.settings.lk_base_c,
            )
            room_c, slab_c = (
                float(room_next) + float(self.extra_drift[t]),
                float(slab_next),
            )
            hours.append(
                self.hour_plan(t, rts_c, room_c, slab_c, {"binding": "baseline"})
            )
        return hours
