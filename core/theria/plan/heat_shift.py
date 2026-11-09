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
    ZoneCurve,
    cooling_c_per_h,
    next_delta_c,
    next_delta_closed_c,
)
from core.theria.plan.planner import HourPlan, Plan, PlanInputs, _Problem, plan
from core.theria.plan.pump_model import PumpParams, flow_target
from core.theria.plan.room_model import RoomParams, valve_open
from core.theria.plan.settings import PlanSettings
from core.theria.plan.shift_rules import (
    CLOSING,
    NORMAL,
    OPENING,
    ShiftWindow,
    balanced,
    hold_window,
    thermostat_modes,
    thermostat_target,
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


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], values.astype(int), [0]])))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _forecast_deltas(
    problem: _Problem,
    baseline: list[HourPlan],
    adjust: np.ndarray,
    modes: list[str],
) -> dict[ZoneCurve, list[float]]:
    """What the setpoint and the thermostats do to each zone against the flat setpoint, per hour."""
    s = problem.settings
    deltas = dict.fromkeys(ZONES, 0.0)
    out: dict[ZoneCurve, list[float]] = {zone: [] for zone in ZONES}
    for t, flat in enumerate(baseline):
        rts = _setpoint(s, adjust[t])
        t_out = problem.inputs.t_out_c[t]
        pump_delta = float(
            flow_target(problem.pump, t_out, rts)
            - flow_target(problem.pump, t_out, s.rts_base_c)
        )
        base_flow = problem.floor_flow(t, s.rts_base_c)
        target = s.lk_upper_c if modes[t] == OPENING else s.lk_setpoint_c
        open_share = float(valve_open(flat.room_c_end, target))
        deltas[MAIN] = (
            next_delta_closed_c(
                MAIN,
                deltas[MAIN],
                float(valve_open(flat.room_c_end, s.lk_setpoint_c)),
                base_flow - flat.room_c_end,
            )
            if modes[t] == CLOSING
            else next_delta_c(
                MAIN,
                deltas[MAIN],
                problem.floor_flow(t, rts) - base_flow,
                open_share,
            )
        )
        deltas[SALON] = next_delta_c(SALON, deltas[SALON], pump_delta)
        for zone in ZONES:
            out[zone].append(deltas[zone])
    return out


def _setpoint(settings: PlanSettings, adjustment: float) -> float:
    return float(
        np.clip(
            settings.rts_base_c + adjustment, settings.rts_min_c, settings.rts_max_c
        )
    )


def _worth_storing(
    problem: _Problem,
    baseline: list[HourPlan],
    adjust: np.ndarray,
    window: int,
) -> np.ndarray:
    """The adjustments with only the buys whose heat is needed at the sell.

    A buy stands only if it ends just before a sell, some room would end that sell under its closing
    target without the buy, that room can be opened, and the raise reaches the floor. A sell that has
    lost its buy must still beat the price the heat is bought back at afterwards.
    """
    s = problem.settings
    inputs = problem.inputs
    offsets = (
        inputs.shift.room_offsets_c
        if inputs.shift is not None and inputs.shift.room_offsets_c
        else (0.0,)
    )
    sells = _runs(adjust < 0)
    no_buys = np.minimum(adjust, 0.0)
    modes = thermostat_modes(no_buys, inputs.hour_starts, None, s.shift_buy_max_gap_h)
    rooms = [
        flat.room_c_end + delta
        for flat, delta in zip(
            baseline,
            _forecast_deltas(problem, baseline, no_buys, modes)[MAIN],
            strict=True,
        )
    ]
    out = no_buys.copy()
    served: set[int] = set()
    for start, end in _runs(adjust > 0):
        following = next((k for k, sell in enumerate(sells) if sell[0] >= end), None)
        if following is None or sells[following][0] - end > s.shift_buy_max_gap_h:
            continue
        last = sells[following][1] - 1
        night = s.in_night(inputs.hour_starts[last])
        closing_target = thermostat_target(CLOSING, night, None, s)
        hour_night = s.in_night(inputs.hour_starts[start])
        needed = any(
            rooms[last] + offset < closing_target
            and thermostat_target(OPENING, hour_night, rooms[start] + offset, s)
            == s.lk_upper_c
            for offset in offsets
        )
        if not needed:
            continue
        for t in range(start, end):
            raised = _setpoint(s, adjust[t])
            if problem.floor_flow(t, raised) > problem.floor_flow(t, s.rts_base_c):
                out[t] = adjust[t]
                served.add(following)
    prices = inputs.price_sek_per_kwh
    for k, (start, end) in enumerate(sells):
        after = prices[end : end + window]
        if k in served or len(after) == 0:
            continue
        if (
            float(prices[start:end].mean() - after.mean())
            <= s.shift_min_gain_sek_per_kwh
        ):
            out[start:end] = 0.0
    return np.asarray(out)


def plan_shift(
    inputs: PlanInputs, settings: PlanSettings, pump: PumpParams, room: RoomParams
) -> Plan:
    """The base setpoint, lowered in the dearest windows, raised before them where the heat is needed."""
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
    baseline = problem.flat_plan(settings.rts_base_c)
    if settings.plan_closes_valves:
        adjust = _worth_storing(problem, baseline, adjust, window)
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
    deltas = _forecast_deltas(problem, baseline, adjust, modes)
    hours = []
    for t, flat in enumerate(baseline):
        rts = _setpoint(settings, adjust[t])
        closed = modes[t] == CLOSING
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
        hour = problem.hour_plan(
            t,
            rts,
            flat.room_c_end + deltas[MAIN][t],
            flat.slab_c_end,
            reason,
            tail=False,
            closed=closed,
        )
        energy = hour.energy_kwh
        if closed:
            # closing saves nothing on loops that would have been closed anyway
            share = float(valve_open(flat.room_c_end, settings.lk_setpoint_c))
            energy = problem.energy_kwh(t, rts) * (
                1.0 - (1.0 - settings.closed_share) * share
            )
        hours.append(
            replace(
                hour,
                energy_kwh=energy,
                energy_cost_sek=hour.price_sek_per_kwh * energy,
                thermostat=modes[t],
                main_delta_c=deltas[MAIN][t],
                salon_delta_c=deltas[SALON][t],
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
