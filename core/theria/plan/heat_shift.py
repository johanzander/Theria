"""Moving heating from dear to cheap hours, after PowerSaver's heat capacitor.

The dear and cheap windows are found as in PowerSaver: slide a window over the prices, take the local
minima as buys and the local maxima as sells, drop the pairs whose price difference is not worth it.
The length of the windows comes from the house (`house_model`): how long the zone that cools fastest can
coast before it has lost the allowed swing. Unlike PowerSaver the setpoint returns to the base after a
window: the pump has no room sensor, so a raised setpoint keeps costing energy as long as it stands.

The windows are found from local midnight, so that a replan inside a window finds the same window.
"""

from dataclasses import replace
from typing import Any

import numpy as np

from core.theria.plan.house_model import (
    MAIN,
    SALON,
    ZONES,
    cooling_c_per_h,
    next_delta_c,
    next_delta_closed_c,
)
from core.theria.plan.planner import Plan, PlanInputs, _Problem, plan
from core.theria.plan.pump_model import PumpParams, flow_target
from core.theria.plan.room_model import RoomParams, valve_open
from core.theria.plan.settings import PlanSettings
from core.theria.plan.shift_rules import (
    CLOSING,
    NORMAL,
    ShiftWindow,
    balanced,
    hold_window,
    thermostat_modes,
)

MIN_WINDOW_H = 1
MAX_WINDOW_H = 8

Trade = tuple[int, int]


def shift_window_hours(t_out_c: float, main_room_c: float, swing_c: float) -> int:
    """How many hours the zone that cools fastest takes to go from +swing to -swing."""
    fastest = max(
        cooling_c_per_h(MAIN, main_room_c, t_out_c),
        cooling_c_per_h(SALON, SALON.reference_c, t_out_c),
    )
    if fastest <= 0.0:
        return MAX_WINDOW_H
    return int(min(MAX_WINDOW_H, max(MIN_WINDOW_H, round(2.0 * swing_c / fastest))))


def _sliding_means(prices: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(prices, np.ones(window) / window, mode="valid")


def _turning_points(means: np.ndarray, window: int) -> tuple[list[int], list[int]]:
    """Local minima (buys) and the local maxima (sells) after each of them."""
    n = len(means)
    spacing = (window + 1) // 2
    buys: list[int] = []
    sells: list[int] = []
    i = 0
    while i < n - 1:
        while i < n - 1 and means[i + 1] < means[i]:
            i += 1
        if i == n - 1:
            break
        buys.append(i)
        i += spacing
        while i < n and means[i] >= means[i - 1]:
            i += 1
        sells.append(min(i - 1, n - 1))
        i += spacing
    return buys, sells


def find_trades(prices: object, window: int, min_gain: float) -> list[Trade]:
    """(buy start, sell start) pairs whose price difference is above `min_gain` per kWh."""
    series = np.asarray(prices, dtype=float)
    if window > len(series):
        return []
    means = _sliding_means(series, window)
    buys, sells = _turning_points(means, window)
    while buys:
        steps = [(i, i, means[sells[i]] - means[buys[i]]) for i in range(len(buys))] + [
            (i, i - 1, means[sells[i - 1]] - means[buys[i]])
            for i in range(1, len(buys))
        ]
        buy, sell, gain = min(steps, key=lambda step: step[2])
        if gain > min_gain:
            break
        del buys[buy], sells[sell]
    return list(zip(buys, sells, strict=True))


def _adjustments(n: int, trades: list[Trade], window: int, step_c: float) -> np.ndarray:
    """Setpoint change per hour; a sell wins where a window overlaps a buy."""
    adjust = np.zeros(n)
    for buy, _ in trades:
        adjust[buy : buy + window] = step_c
    for _, sell in trades:
        adjust[sell : sell + window] = -step_c
    return adjust


def plan_shift(
    inputs: PlanInputs, settings: PlanSettings, pump: PumpParams, room: RoomParams
) -> Plan:
    """The base setpoint, lowered in the dearest windows and raised in the cheap ones before them."""
    problem = _Problem(inputs, settings, pump, room)
    past = (
        np.zeros(0)
        if inputs.past_prices_sek_per_kwh is None
        else np.asarray(inputs.past_prices_sek_per_kwh, dtype=float)
    )
    series = np.concatenate([past, inputs.price_sek_per_kwh])
    window = shift_window_hours(
        float(np.mean(inputs.t_out_c)),
        inputs.initial.room_c,
        settings.shift_room_swing_c,
    )
    trades = find_trades(series, window, settings.shift_min_gain_sek_per_kwh)
    adjust = balanced(
        _adjustments(len(series), trades, window, settings.shift_rts_c), series
    )[len(past) :]
    committed: ShiftWindow | None = None
    if inputs.shift is not None:
        adjust, committed = hold_window(adjust, inputs.hour_starts, inputs.shift)
    modes = (
        thermostat_modes(
            adjust, inputs.hour_starts, committed, settings.shift_buy_max_gap_h
        )
        if settings.plan_closes_valves
        else [NORMAL] * len(adjust)
    )
    baseline = problem.flat_plan(settings.rts_base_c)
    deltas = dict.fromkeys(ZONES, 0.0)
    hours = []
    for t, flat in enumerate(baseline):
        rts = float(
            np.clip(
                settings.rts_base_c + adjust[t], settings.rts_min_c, settings.rts_max_c
            )
        )
        t_out = inputs.t_out_c[t]
        pump_delta = float(
            flow_target(pump, t_out, rts)
            - flow_target(pump, t_out, settings.rts_base_c)
        )
        floor_delta = problem.floor_flow(t, rts) - problem.floor_flow(
            t, settings.rts_base_c
        )
        closed = modes[t] == CLOSING
        deltas[MAIN] = (
            next_delta_closed_c(
                MAIN,
                deltas[MAIN],
                float(valve_open(flat.room_c_end, settings.lk_setpoint_c)),
                problem.floor_flow(t, settings.rts_base_c) - flat.room_c_end,
            )
            if closed
            else next_delta_c(MAIN, deltas[MAIN], floor_delta)
        )
        deltas[SALON] = next_delta_c(SALON, deltas[SALON], pump_delta)
        guard = (
            committed.ended_by
            if committed is not None and flat.start < committed.end
            else None
        )
        reason: dict[str, Any] = {
            "binding": (
                "shift_guard"
                if guard is not None
                else (
                    "shift_sell"
                    if adjust[t] < 0
                    else ("shift_buy" if adjust[t] > 0 else "shift_none")
                )
            ),
            "window_hours": window,
        }
        if guard is not None:
            reason["guard"] = guard
        hours.append(
            replace(
                problem.hour_plan(
                    t,
                    rts,
                    flat.room_c_end + deltas[MAIN],
                    flat.slab_c_end,
                    reason,
                    tail=False,
                    closed=closed,
                ),
                thermostat=modes[t],
                main_delta_c=deltas[MAIN],
                salon_delta_c=deltas[SALON],
            )
        )
    violations = sum(
        1 for h in hours if h.room_c_end + h.sun_swing_c < h.floor_c - 0.05
    )
    return Plan(
        hours,
        baseline,
        float(sum(h.energy_cost_sek + h.comfort_cost_sek for h in hours)),
        float(sum(h.energy_cost_sek + h.comfort_cost_sek for h in baseline)),
        violations,
        end_reference_c=float(problem.targets[-1]),
        end_cost_sek_per_c=0.0,
        shift_window=committed,
    )


def make_plan(
    inputs: PlanInputs, settings: PlanSettings, pump: PumpParams, room: RoomParams
) -> Plan:
    """The plan by the chosen strategy."""
    if settings.plan_strategy == "heat_shift":
        return plan_shift(inputs, settings, pump, room)
    return plan(inputs, settings, pump, room)
