"""The whole house's heat need, fitted to the pump's heat counter.

Only the whole house has a measure of heat: the pump's counter of the heat it supplied to the heating
(hot water excluded). Per day that heat is what the house lost less what other sources gave:

    heat = H (room - outdoor) - F

H is the whole house's loss in W/K, comparable with the zones' figures from the heat demand analysis;
F is the free heat in kW (household electricity, people, sun and the stove, as a mean). The room is
the ground floor's LK rooms, the only indoor temperature with a long history. Whole days smooth out
the floor's storage and the pump's cycles. See docs/ALGORITHM_EXPLAINED.md.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.series import FloatArray

HOUSE_FIT_DAYS = 365
MAX_OUTDOOR_C = 12.0  # milder days are mostly hot water, sun and the summer stop
MIN_HOURS_PER_DAY = 20
MIN_DAYS = 30
H_PLAUSIBLE_W_PER_K = (200.0, 1500.0)
FREE_PLAUSIBLE_KW = (-1.0, 6.0)


@dataclass(frozen=True)
class HouseLoss:
    w_per_k: float  # H, the whole house's loss
    free_kw: float  # F, the mean free heat
    days: int  # whole days the fit was made on
    rmse_kw: float  # how far a day's mean heat is from the fit


def _whole_days(values: FloatArray, n_days: int) -> FloatArray:
    return values[-n_days * 24 :].reshape(n_days, 24)


def fit_house_loss(
    hours: Sequence[datetime],
    heat_kwh: FloatArray,
    room_c: FloatArray,
    t_out_c: FloatArray,
) -> HouseLoss:
    """Fit H and F on the last year's whole days below MAX_OUTDOOR_C; ModelFitError when it cannot."""
    n_days = min(HOUSE_FIT_DAYS, len(hours) // 24)
    if n_days == 0:
        raise ModelFitError("house loss needs at least one whole day of history")
    heat = _whole_days(heat_kwh, n_days)
    room = _whole_days(room_c, n_days)
    out = _whole_days(t_out_c, n_days)
    known = np.isfinite(heat) & np.isfinite(room) & np.isfinite(out)
    full = known.sum(axis=1) >= MIN_HOURS_PER_DAY
    # means over the hours with all three, so that a day is compared with itself
    count = np.maximum(known.sum(axis=1), 1)

    def day_mean(values: FloatArray) -> FloatArray:
        means: FloatArray = np.where(known, values, 0.0).sum(axis=1) / count
        return means

    heat_kw = day_mean(heat)
    delta = day_mean(room - out)
    rows = full & (day_mean(out) < MAX_OUTDOOR_C)
    if int(rows.sum()) < MIN_DAYS:
        raise ModelFitError(
            f"house loss needs {MIN_DAYS} whole days below {MAX_OUTDOOR_C} C with heat, "
            f"room and outdoor, found {int(rows.sum())}"
        )
    design = np.column_stack([delta[rows], -np.ones(int(rows.sum()))])
    (h_kw, free_kw), *_ = np.linalg.lstsq(design, heat_kw[rows], rcond=None)
    w_per_k = float(h_kw) * 1000.0
    if not H_PLAUSIBLE_W_PER_K[0] <= w_per_k <= H_PLAUSIBLE_W_PER_K[1]:
        raise ModelFitError(f"implausible whole-house loss {w_per_k:.0f} W/K")
    if not FREE_PLAUSIBLE_KW[0] <= free_kw <= FREE_PLAUSIBLE_KW[1]:
        raise ModelFitError(f"implausible free heat {free_kw:.2f} kW")
    residual = heat_kw[rows] - design @ np.array([h_kw, free_kw])
    return HouseLoss(
        w_per_k=w_per_k,
        free_kw=float(free_kw),
        days=int(rows.sum()),
        rmse_kw=float(np.sqrt(np.mean(residual**2))),
    )


def house_need_kw(house: HouseLoss, room_c: float, t_out_c: float) -> float | None:
    """What the whole house loses at this room and outdoor temperature, whatever meets it."""
    if not (math.isfinite(room_c) and math.isfinite(t_out_c)):
        return None
    return max(0.0, house.w_per_k * (room_c - t_out_c) / 1000.0)


def house_pump_kw(house: HouseLoss, room_c: float, t_out_c: float) -> float | None:
    """What the pump is expected to supply: the loss less the free heat."""
    need = house_need_kw(house, room_c, t_out_c)
    return None if need is None else max(0.0, need - house.free_kw)
