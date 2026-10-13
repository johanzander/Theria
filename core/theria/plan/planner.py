"""Dynamic-programming planner: choose the room setpoint for each coming hour."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt

from core.theria.exceptions import OptimizationError
from core.theria.plan.pump_model import PumpParams, flow_target, total_kwh_per_hour
from core.theria.plan.room_model import RoomParams, RoomState, step
from core.theria.plan.settings import PlanSettings

FloatArray = npt.NDArray[np.float64]
FloatGrid = npt.NDArray[np.floating[Any]]  # cost-to-go tables are float32

ROOM_MARGIN_C = (
    3.0  # room grid reaches this far below the lowest floor / above the ceiling
)
SLAB_MARGIN_C = 2.0
ROOM_BIN_C = 0.1
SLAB_BIN_C = 1.0
VIOLATION_COST_SEK_PER_C_HOUR = 100.0  # floor / ceiling are effectively hard
SWITCH_COST_SEK_PER_C = 0.02  # tie-breaker: prefer smooth plans, avoids pump jitter
MAX_TERMINAL_HOURS = 168.0  # cap on how long an end-of-horizon deficit is charged
FLOW_CAP_PENALTY_SEK = (
    1e6  # makes setpoints above the flow cap unattractive without inf/nan
)
TAIL_HOURS = 6  # last hours of a horizon are shown but flagged low-confidence


@dataclass(frozen=True)
class PlanInputs:
    """Hourly forcing for the horizon. Arrays have one entry per hour."""

    hour_starts: list[datetime]
    price_sek_per_kwh: FloatArray
    extreme: npt.NDArray[np.bool_]
    t_out_c: FloatArray
    sun_wm2: FloatArray
    initial: RoomState
    previous_rts_c: float
    # What the room does on its own, per hour (the learned night cooling and the warming by day
    # that balances it); None when nothing is known and the anchored drift is all there is.
    extra_drift_c_per_h: FloatArray | None = None


@dataclass(frozen=True)
class HourPlan:
    start: datetime
    rts_c: float
    price_sek_per_kwh: float
    extreme: bool
    floor_c: float
    flow_target_c: float
    energy_kwh: float
    room_c_end: float
    slab_c_end: float
    energy_cost_sek: float
    comfort_cost_sek: float
    tail: bool
    reason: dict[str, Any]


@dataclass(frozen=True)
class Plan:
    hours: list[HourPlan]
    baseline_hours: list[HourPlan]  # same model, flat rts_base
    objective_sek: float
    baseline_objective_sek: float
    floor_violation_hours: int


def _interp2(grid: FloatGrid, r_idx: FloatArray, s_idx: FloatArray) -> FloatArray:
    """Bilinear lookup of `grid` (NR, NS) at fractional indices, clipped to the grid."""
    r_idx = np.clip(r_idx, 0, grid.shape[0] - 1.000001)
    s_idx = np.clip(s_idx, 0, grid.shape[1] - 1.000001)
    r0 = np.floor(r_idx).astype(int)
    s0 = np.floor(s_idx).astype(int)
    fr = r_idx - r0
    fs = s_idx - s0
    return np.asarray(
        grid[r0, s0] * (1 - fr) * (1 - fs)
        + grid[r0 + 1, s0] * fr * (1 - fs)
        + grid[r0, s0 + 1] * (1 - fr) * fs
        + grid[r0 + 1, s0 + 1] * fr * fs,
        dtype=float,
    )


def sun_filtered_sequence(
    room: RoomParams, initial: float, sun: FloatArray
) -> FloatArray:
    """Filtered sun at the start of each hour (exogenous, so not a DP state)."""
    out = np.empty(len(sun))
    value = initial
    k = 1.0 - np.exp(-1.0 / room.sun_lag_h)
    for i, s in enumerate(sun):
        out[i] = value
        value = value + (s - value) * k
    return out


class _Problem:
    """Everything the DP needs for one horizon, with the model transition and costs."""

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
        # A room that ends the horizon below target stays below it for about the room's time
        # constant, not for a few hours: charge the deficit for that long.
        # ... and the horizon must end where it started: a sag has to be paid back inside the
        # known prices (like the battery's state of charge in the BESS planner), otherwise
        # every replan would start a little lower than the last one.
        self.reference_room_c = max(inputs.initial.room_c, settings.target_c)
        self.terminal_hours = min(MAX_TERMINAL_HOURS, 1.0 / (room.k_slab + room.k_out))
        self.actions = np.arange(
            settings.rts_min_c, settings.rts_max_c + 1e-9, settings.rts_step_c
        )
        self.extra_drift = (
            np.zeros(self.horizon)
            if inputs.extra_drift_c_per_h is None
            else np.asarray(inputs.extra_drift_c_per_h, dtype=float)
        )
        self.sun_f = sun_filtered_sequence(
            room, inputs.initial.sun_filtered_wm2, inputs.sun_wm2
        )
        # The room is judged at the end of each hour, so the night ends (and comfort is due)
        # when the state at 06:00 is reached.
        night = np.array(
            [
                settings.in_night(start + timedelta(hours=1))
                for start in inputs.hour_starts
            ],
            dtype=bool,
        )
        self.targets = np.where(night, settings.night_target_c, settings.target_c)
        self.floors = np.where(
            inputs.extreme,
            settings.extreme_floor_c,
            np.where(night, settings.night_floor_c, settings.floor_c),
        )
        # Setpoints whose flow target exceeds the floor limit are not allowed; the lowest
        # setpoint always is, so that a plan exists.
        flows = flow_target(pump, inputs.t_out_c[:, None], self.actions[None, :])
        baseline_flow = flow_target(pump, inputs.t_out_c, settings.rts_base_c)
        # never above the floor limit; no boost above the range where the electric heater
        # starts, unless the baseline itself already is there (cold weather): then cut only
        cap = np.minimum(
            settings.max_flow_c, np.maximum(settings.boost_flow_limit_c, baseline_flow)
        )
        self.allowed: npt.NDArray[np.bool_] = np.asarray(
            flows <= cap[:, None], dtype=bool
        )
        self.allowed[:, 0] = True
        # Grids cover every state the models can reach: the hottest flow temperature
        # (coldest hour at the highest setpoint) and the comfort limits.
        max_flow = float(np.max(flow_target(pump, inputs.t_out_c, settings.rts_max_c)))
        self.room_grid = np.arange(
            np.floor(settings.extreme_floor_c - ROOM_MARGIN_C),
            np.ceil(settings.ceiling_c + ROOM_MARGIN_C) + 1e-9,
            ROOM_BIN_C,
        )
        self.slab_grid = np.arange(
            np.floor(min(settings.extreme_floor_c, float(np.min(inputs.t_out_c)))),
            np.ceil(max(max_flow, inputs.initial.slab_c) + SLAB_MARGIN_C) + 1e-9,
            SLAB_BIN_C,
        )

    def transition(
        self, t: int, a: int, room_c: FloatArray, slab_c: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Room and slab temperature at the end of hour t for action index a."""
        flow = float(flow_target(self.pump, self.inputs.t_out_c[t], self.actions[a]))
        room_next, slab_next, _ = step(
            self.room,
            room_c,
            slab_c,
            self.sun_f[t],
            flow,
            self.inputs.t_out_c[t],
            self.inputs.sun_wm2[t],
            self.settings.lk_setpoint_c,
        )
        return (
            np.asarray(room_next, dtype=float) + self.extra_drift[t],
            np.asarray(slab_next, dtype=float),
        )

    def comfort_cost(
        self, room_c: FloatArray, floor_c: float, target_c: float
    ) -> FloatArray:
        s = self.settings
        return np.asarray(
            s.comfort_cost_sek_per_c_hour * np.maximum(0.0, target_c - room_c)
            + VIOLATION_COST_SEK_PER_C_HOUR * np.maximum(0.0, floor_c - room_c)
            + VIOLATION_COST_SEK_PER_C_HOUR * np.maximum(0.0, room_c - s.ceiling_c),
            dtype=float,
        )

    def energy_kwh(self, t: int, rts_c: float) -> float:
        return float(total_kwh_per_hour(self.pump, self.inputs.t_out_c[t], rts_c))

    def stage_cost(self, t: int, a: int, room_next: FloatArray) -> FloatArray:
        energy_cost = float(self.inputs.price_sek_per_kwh[t]) * self.energy_kwh(
            t, float(self.actions[a])
        )
        other_rooms = self.settings.setpoint_deviation_cost_sek_per_c_hour * abs(
            float(self.actions[a]) - self.settings.rts_base_c
        )
        return (
            energy_cost
            + other_rooms
            + self.comfort_cost(
                room_next, float(self.floors[t]), float(self.targets[t])
            )
        )

    def q_value(
        self,
        t: int,
        a: int,
        room_c: FloatArray,
        slab_c: FloatArray,
        next_value: FloatGrid,
    ) -> FloatArray:
        """Immediate cost plus interpolated cost-to-go for action a from the given states."""
        room_next, slab_next = self.transition(t, a, room_c, slab_c)
        future = _interp2(
            next_value[:, :, a],
            (room_next - self.room_grid[0]) / ROOM_BIN_C,
            (slab_next - self.slab_grid[0]) / SLAB_BIN_C,
        )
        return self.stage_cost(t, a, room_next) + future

    def solve(self) -> tuple[FloatGrid, npt.NDArray[np.int8]]:
        """Backward induction. Returns cost-to-go and policy, indexed by previous action."""
        s = self.settings
        n_actions = len(self.actions)
        max_move = round(s.max_rts_step_per_hour_c / s.rts_step_c)
        rr, ss = np.meshgrid(self.room_grid, self.slab_grid, indexing="ij")
        shape = (len(self.room_grid), len(self.slab_grid), n_actions)
        value = np.empty((self.horizon + 1, *shape), dtype=np.float32)
        terminal = (
            self.terminal_hours
            * s.comfort_cost_sek_per_c_hour
            * np.maximum(0.0, self.reference_room_c - rr)
        )
        value[self.horizon] = terminal[:, :, None]
        policy = np.empty((self.horizon, *shape), dtype=np.int8)
        for t in range(self.horizon - 1, -1, -1):
            q = np.empty(shape)
            for a in range(n_actions):
                q[:, :, a] = self.q_value(t, a, rr, ss, value[t + 1])
                if not self.allowed[t, a]:
                    q[:, :, a] += FLOW_CAP_PENALTY_SEK
            best = np.full(shape, np.inf)
            best_a = np.zeros(shape, dtype=np.int8)
            # raising the setpoint is limited to max_move steps per hour (the rebound is where
            # the electric heater starts); lowering it is not
            for offset in range(-(n_actions - 1), max_move + 1):
                shifted = np.full(shape, np.inf)
                lo, hi = max(0, -offset), min(n_actions, n_actions - offset)
                shifted[:, :, lo:hi] = q[:, :, lo + offset : hi + offset] + (
                    SWITCH_COST_SEK_PER_C * abs(offset) * s.rts_step_c
                )
                better = shifted < best
                best = np.where(better, shifted, best)
                best_a = np.where(
                    better, (np.arange(n_actions) + offset)[None, None, :], best_a
                ).astype(np.int8)
            value[t] = best
            policy[t] = best_a
        return value, policy

    def hour_plan(
        self,
        t: int,
        rts_c: float,
        room_end: float,
        slab_end: float,
        reason: dict[str, Any],
        tail: bool,
    ) -> HourPlan:
        energy = self.energy_kwh(t, rts_c)
        price = float(self.inputs.price_sek_per_kwh[t])
        floor_c = float(self.floors[t])
        return HourPlan(
            start=self.inputs.hour_starts[t],
            rts_c=rts_c,
            price_sek_per_kwh=price,
            extreme=bool(self.inputs.extreme[t]),
            floor_c=floor_c,
            flow_target_c=float(flow_target(self.pump, self.inputs.t_out_c[t], rts_c)),
            energy_kwh=energy,
            room_c_end=room_end,
            slab_c_end=slab_end,
            energy_cost_sek=price * energy,
            comfort_cost_sek=float(
                self.comfort_cost(
                    np.array([room_end]), floor_c, float(self.targets[t])
                )[0]
            ),
            tail=tail,
            reason=reason,
        )

    def follow_policy(
        self, value: FloatGrid, policy: npt.NDArray[np.int8]
    ) -> list[HourPlan]:
        """Roll the optimal policy forward from the initial state, with explanations."""
        room_c, slab_c = self.inputs.initial.room_c, self.inputs.initial.slab_c
        prev = int(np.argmin(np.abs(self.actions - self.inputs.previous_rts_c)))
        rank = np.argsort(np.argsort(self.inputs.price_sek_per_kwh)) / max(
            1, self.horizon - 1
        )
        hours: list[HourPlan] = []
        for t in range(self.horizon):
            r_i = int(
                np.clip(
                    round((room_c - self.room_grid[0]) / ROOM_BIN_C),
                    0,
                    len(self.room_grid) - 1,
                )
            )
            s_i = int(
                np.clip(
                    round((slab_c - self.slab_grid[0]) / SLAB_BIN_C),
                    0,
                    len(self.slab_grid) - 1,
                )
            )
            a = int(policy[t, r_i, s_i, prev])
            here = (np.array([[room_c]]), np.array([[slab_c]]))
            chosen = float(self.q_value(t, a, *here, value[t + 1])[0, 0])
            marginal = {
                label: float(self.q_value(t, b, *here, value[t + 1])[0, 0]) - chosen
                for label, b in (("minus_step_sek", a - 1), ("plus_step_sek", a + 1))
                if 0 <= b < len(self.actions) and self.allowed[t, b]
            }
            room_next, slab_next = self.transition(t, a, *here)
            room_end, slab_end = float(room_next[0, 0]), float(slab_next[0, 0])
            hours.append(
                self.hour_plan(
                    t,
                    float(self.actions[a]),
                    room_end,
                    slab_end,
                    self.reason(a, room_end, t, float(rank[t]), marginal),
                    tail=t >= self.horizon - TAIL_HOURS,
                )
            )
            room_c, slab_c, prev = room_end, slab_end, a
        return hours

    def flat_plan(self, rts_c: float) -> list[HourPlan]:
        """The same models with a constant room setpoint: the comparison baseline."""
        room_c, slab_c = self.inputs.initial.room_c, self.inputs.initial.slab_c
        hours: list[HourPlan] = []
        for t in range(self.horizon):
            flow = float(flow_target(self.pump, self.inputs.t_out_c[t], rts_c))
            room_next, slab_next, _ = step(
                self.room,
                room_c,
                slab_c,
                self.sun_f[t],
                flow,
                self.inputs.t_out_c[t],
                self.inputs.sun_wm2[t],
                self.settings.lk_setpoint_c,
            )
            room_c, slab_c = float(room_next) + float(self.extra_drift[t]), float(
                slab_next
            )
            hours.append(
                self.hour_plan(
                    t, rts_c, room_c, slab_c, {"binding": "baseline"}, tail=False
                )
            )
        return hours

    def reason(
        self,
        a: int,
        room_end: float,
        t: int,
        price_rank: float,
        marginal: dict[str, float],
    ) -> dict[str, Any]:
        s = self.settings
        floor_c = float(self.floors[t])
        if room_end <= floor_c + 0.05:
            binding = "floor"
        elif room_end >= s.ceiling_c - 0.05:
            binding = "ceiling"
        elif a + 1 < len(self.actions) and not self.allowed[t, a + 1]:
            binding = "flow_cap"
        elif a == 0:
            binding = "rts_min"
        elif a == len(self.actions) - 1:
            binding = "rts_max"
        else:
            binding = "price_trade_off"
        return {
            "binding": binding,
            # 0 = cheapest hour of the horizon, 1 = most expensive
            "price_rank": price_rank,
            "extreme_price": bool(self.inputs.extreme[t]),
            # extra objective if the setpoint were one step lower / higher
            "marginal_objective_sek": marginal,
        }


def plan(
    inputs: PlanInputs, settings: PlanSettings, pump: PumpParams, room: RoomParams
) -> Plan:
    """Backward dynamic programming over (room, slab, previous setpoint)."""
    horizon = len(inputs.hour_starts)
    arrays = (inputs.price_sek_per_kwh, inputs.extreme, inputs.t_out_c, inputs.sun_wm2)
    if horizon < 2 or any(len(a) != horizon for a in arrays):
        raise OptimizationError(
            "planner needs at least 2 hours and equally long inputs"
        )
    problem = _Problem(inputs, settings, pump, room)
    value, policy = problem.solve()
    hours = problem.follow_policy(value, policy)
    baseline = problem.flat_plan(settings.rts_base_c)
    objective = sum(h.energy_cost_sek + h.comfort_cost_sek for h in hours)
    base_objective = sum(h.energy_cost_sek + h.comfort_cost_sek for h in baseline)
    if (
        settings.min_saving_sek > 0
        and base_objective - objective < settings.min_saving_sek
    ):
        hours = [replace(h, reason={"binding": "below_min_saving"}) for h in baseline]
        objective = base_objective
    violations = sum(1 for h in hours if h.room_c_end < h.floor_c - 0.05)
    return Plan(hours, baseline, float(objective), float(base_objective), violations)
