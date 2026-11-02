"""The salon's hourly plan: when to heat and what the thermostats are set to.

The salon is one room with a thermal mass `C` and a loss `U`. Each hour the thermostats hold a level
(the end-of-hour mean) somewhere between what the room does on its own (cooling) and what the radiators
can deliver (`heat_power_w`). A dynamic program over a 0.1 C grid chooses the levels that cost least
at the hourly prices, while the mean stays above the open target plus a margin in opening hours and above
the closed floor otherwise. Heat stored above the target at the end of the horizon is credited at the
horizon's mean price, at most 0.5 C of it.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.theria.plan.lk_executor import LK_STEP_C, step_up
from core.theria.radiators.settings import RadiatorPrefs

GRID_C = 0.1
CREDIT_CAP_C = 0.5
ACTIVE_HEAT_C = (
    0.05  # warmer than the room's own cooling by this much counts as heating
)
MAX_BIAS_C = 2.0
BELOW_BOUND_SEK_PER_C = 100.0  # a soft bound: only the hours that cannot be met pay it
GRID_BELOW_C = 8.0  # how far under the closed floor the grid reaches
SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class SalonModel:
    c_j_per_k: float
    u_w_per_k: float
    heat_power_w: float


@dataclass(frozen=True)
class SalonHour:
    start: datetime
    open: bool
    heat: bool  # the radiators are actively heating this hour
    mean_end_c: float
    setpoint_c: float  # what both thermostats are set to
    price_sek_per_kwh: float
    reason: str


def step_down(value_c: float) -> float:
    """The highest valid thermostat setpoint that is not above `value_c`."""
    return math.floor(value_c / LK_STEP_C + 1e-9) * LK_STEP_C


def _transition_cost(
    model: SalonModel,
    t_from: np.ndarray,
    grid: np.ndarray,
    t_out_c: float,
    price: float,
    bound_c: float,
    preheat_c: float,
) -> np.ndarray:
    """SEK of going from each `t_from` to each grid level in an hour; infinite when not possible.

    Ending below the hour's bound is allowed at a high price per degree, so a single hour that cannot be
    met does not decide the other hours. Heating above the pre-heat level is not allowed, but a room that
    is already above it may hold or cool.
    """
    loss_w = model.u_w_per_k * (t_from - t_out_c)
    seconds = SECONDS_PER_HOUR / model.c_j_per_k
    t_down = t_from - loss_w * seconds
    t_up = t_from + (model.heat_power_w - loss_w) * seconds
    to = grid[None, :]
    frm = t_from[:, None]
    reachable = (to >= t_down[:, None] - 1e-9) & (to <= t_up[:, None] + 1e-9)
    reachable &= (to <= preheat_c + 1e-9) | (to <= frm + 1e-9)
    wh = model.c_j_per_k * (to - frm) / SECONDS_PER_HOUR + model.u_w_per_k * (
        (frm + to) / 2 - t_out_c
    )
    cost = price * np.maximum(wh, 0.0) / 1000.0
    cost = cost + BELOW_BOUND_SEK_PER_C * np.maximum(bound_c - to, 0.0)
    return np.where(reachable, cost, np.inf)


def _solve(
    prices: Sequence[float],
    t_out_c: Sequence[float],
    bounds: Sequence[float],
    start_mean_c: float,
    grid: np.ndarray,
    prefs: RadiatorPrefs,
    model: SalonModel,
) -> list[float] | None:
    """The cheapest end-of-hour levels, or None when the bounds cannot be met."""
    n = len(prices)
    mean_price = float(np.mean(prices))
    ref = prefs.salon_open_target_c + prefs.salon_margin_c
    stored = ref - np.minimum(grid, ref + CREDIT_CAP_C)
    value = [np.empty(0)] * (n + 1)
    value[n] = mean_price * model.c_j_per_k * stored / 3.6e6
    for k in range(n - 1, 0, -1):
        step = _transition_cost(
            model, grid, grid, t_out_c[k], prices[k], bounds[k], prefs.salon_preheat_c
        )
        value[k] = (step + value[k + 1][None, :]).min(axis=1)
    levels: list[float] = []
    here = np.array([start_mean_c])
    for k in range(n):
        step = _transition_cost(
            model, here, grid, t_out_c[k], prices[k], bounds[k], prefs.salon_preheat_c
        )
        total = step[0] + value[k + 1]
        best = int(np.argmin(total))
        if not np.isfinite(total[best]):
            return None
        levels.append(float(grid[best]))
        here = grid[best : best + 1]
    return levels


def _path_cost(
    levels: Sequence[float],
    prices: Sequence[float],
    t_out_c: Sequence[float],
    bounds: Sequence[float],
    start_mean_c: float,
    prefs: RadiatorPrefs,
    model: SalonModel,
) -> float:
    ref = prefs.salon_open_target_c + prefs.salon_margin_c
    total, here = 0.0, start_mean_c
    for level, price, t_out, bound in zip(levels, prices, t_out_c, bounds, strict=True):
        total += BELOW_BOUND_SEK_PER_C * max(bound - level, 0.0)
        wh = model.c_j_per_k * (level - here) / SECONDS_PER_HOUR + model.u_w_per_k * (
            (here + level) / 2 - t_out
        )
        total += price * max(wh, 0.0) / 1000.0
        here = level
    stored = ref - min(here, ref + CREDIT_CAP_C)
    return total + float(np.mean(prices)) * model.c_j_per_k * stored / 3.6e6


def plan_salon(
    hours: list[datetime],
    prices: Sequence[float],
    t_out_c: Sequence[float],
    open_flags: Sequence[bool],
    start_mean_c: float,
    prefs: RadiatorPrefs,
    model: SalonModel,
    thermostat_bias_c: float,
    min_saving_sek: float = 0.2,
) -> list[SalonHour]:
    n = len(hours)
    bias = min(max(thermostat_bias_c, 0.0), MAX_BIAS_C)
    known = [p for p in prices if not math.isnan(p)]
    mean_price = float(np.mean(known)) if known else 1.0
    filled = [mean_price if math.isnan(p) else p for p in prices]
    open_c = prefs.salon_open_target_c + prefs.salon_margin_c
    required = [open_c if o else prefs.salon_closed_floor_c for o in open_flags]
    # the hour before an opening must already end at the open level
    bounds = [max(required[k], required[min(k + 1, n - 1)]) for k in range(n)]
    top = max(prefs.salon_preheat_c, start_mean_c)
    low = math.floor(
        (min(prefs.salon_closed_floor_c, start_mean_c) - GRID_BELOW_C) / GRID_C
    )
    grid = np.arange(low * GRID_C, math.ceil(top / GRID_C) * GRID_C + 1e-9, GRID_C)
    flat_prices = [mean_price] * n
    flat = _solve(flat_prices, t_out_c, bounds, start_mean_c, grid, prefs, model)
    best = _solve(filled, t_out_c, bounds, start_mean_c, grid, prefs, model)
    if flat is None or best is None:
        return _infeasible(
            hours, prices, t_out_c, open_flags, start_mean_c, prefs, model
        )
    saving = _path_cost(
        flat, filled, t_out_c, bounds, start_mean_c, prefs, model
    ) - _path_cost(best, filled, t_out_c, bounds, start_mean_c, prefs, model)
    is_flat = saving < min_saving_sek
    levels = flat if is_flat else best
    out: list[SalonHour] = []
    here = start_mean_c
    for k, level in enumerate(levels):
        cooled = here - model.u_w_per_k * (here - t_out_c[k]) * (
            SECONDS_PER_HOUR / model.c_j_per_k
        )
        heat = level > cooled + ACTIVE_HEAT_C
        if is_flat:
            reason = "flat: prices too flat to move heat, schedule only"
        elif level > here + ACTIVE_HEAT_C and filled[k] < mean_price:
            reason = "pre-heat while cheap"
        elif level > here + ACTIVE_HEAT_C:
            reason = f"heating up to {level:.1f} C"
        elif heat:
            reason = f"holding {level:.1f} C"
        elif level < here - ACTIVE_HEAT_C:
            reason = "coast"
        else:
            reason = "hold"
        if math.isnan(prices[k]):
            reason = f"no price, planned at the mean; {reason}"
        setpoint = min(
            prefs.salon_max_setpoint_c,
            max(prefs.salon_closed_floor_c, step_up(level + bias)),
        )
        if (
            level < bounds[k] - 1e-9
        ):  # this hour cannot be met: all the radiators can give
            heat, setpoint = True, prefs.salon_max_setpoint_c
            reason = "infeasible: heating at full power"
        out.append(
            SalonHour(hours[k], open_flags[k], heat, level, setpoint, prices[k], reason)
        )
        here = level
    return out


def _infeasible(
    hours: list[datetime],
    prices: Sequence[float],
    t_out_c: Sequence[float],
    open_flags: Sequence[bool],
    start_mean_c: float,
    prefs: RadiatorPrefs,
    model: SalonModel,
) -> list[SalonHour]:
    """The bounds cannot be met: heat at full power every hour and say so."""
    out: list[SalonHour] = []
    here = start_mean_c
    seconds = SECONDS_PER_HOUR / model.c_j_per_k
    for k, start in enumerate(hours):
        here += (model.heat_power_w - model.u_w_per_k * (here - t_out_c[k])) * seconds
        out.append(
            SalonHour(
                start,
                open_flags[k],
                True,
                here,
                prefs.salon_max_setpoint_c,
                prices[k],
                "infeasible: heating at full power",
            )
        )
    return out
