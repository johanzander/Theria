"""How much the sun lifts the room above its usual level, from the hours it shines.

The room is slow (days), the sun is quick: a sunny afternoon lifts the room a few degrees, which
is gone again by the next morning. The sun is therefore not a heating rate in the slow model but a
*swing* on top of the level: `room = level + swing_c_per_wm2 * filtered sun`. The swing is read from
each day: the room's temperature in the sunny hours minus its mean in the same day's dark hours (the
level), against the irradiation smoothed with a lag (the room peaks a few hours after the sun).
The gain and the lag with the smallest error are kept. It is refused when there are too few sunny
hours, when it does not reduce the error, or when it is larger than a room can plausibly take.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from core.theria.exceptions import ModelFitError

FloatArray = npt.NDArray[np.float64]

LAGS_H = (1.0, 2.0, 3.0, 4.0)
SUNNY_WM2 = 100.0  # filtered irradiation that counts as a sunny hour
DARK_WM2 = 20.0  # at most this is a dark hour (the day's level is read from these)
MIN_SUNNY_HOURS = 80
MIN_DARK_HOURS_PER_DAY = 6
MAX_OUTDOOR_C = 15.0  # warmer hours are not heating hours
MAX_SWING_C_PER_WM2 = 0.03  # 3 C at 100 W/m2 filtered irradiation


@dataclass(frozen=True)
class SunFit:
    swing_c_per_wm2: float
    lag_h: float
    sunny_hours: int
    rmse_before_c: float  # per hour, without the sun
    rmse_after_c: float


def _filtered(sun: FloatArray, lag_h: float) -> FloatArray:
    out = np.empty(len(sun))
    value, k = float(sun[0]), 1.0 - np.exp(-1.0 / lag_h)
    for i, s in enumerate(sun):
        out[i] = value
        value += (s - value) * k
    return out


def fit_sun(
    hours: list[datetime],
    room_c: FloatArray,
    t_out_c: FloatArray,
    sun_wm2: FloatArray,
) -> SunFit:
    sun = np.nan_to_num(sun_wm2)
    by_day: dict[object, list[int]] = defaultdict(list)
    for i, moment in enumerate(hours):
        by_day[moment.date()].append(i)
    best: tuple[float, float, float, FloatArray, int] | None = None  # error, swing, lag, residual, hours
    for lag in LAGS_H:
        filtered = _filtered(sun, lag)
        y = np.full(len(hours), np.nan)
        for indices in by_day.values():
            idx = np.array(indices)
            dark = idx[
                (filtered[idx] < DARK_WM2)
                & np.isfinite(room_c[idx])
                & (t_out_c[idx] <= MAX_OUTDOOR_C)
            ]
            if len(dark) >= MIN_DARK_HOURS_PER_DAY:
                y[idx] = room_c[idx] - float(np.mean(room_c[dark]))
        usable = np.isfinite(y) & (filtered > DARK_WM2) & (t_out_c <= MAX_OUTDOOR_C)
        sunny = int(np.sum(usable & (filtered >= SUNNY_WM2)))
        if sunny < MIN_SUNNY_HOURS:
            raise ModelFitError(
                f"need {MIN_SUNNY_HOURS} sunny hours to judge the sun, found {sunny}"
            )
        x, r = filtered[usable], y[usable]
        swing = max(float(np.sum(x * r) / np.sum(x * x)), 0.0)
        error = float(np.sqrt(np.mean((r - swing * x) ** 2)))
        if best is None or error < best[0]:
            best = (error, swing, lag, r, sunny)
    assert best is not None
    error, swing, lag, r, sunny = best
    if swing > MAX_SWING_C_PER_WM2:
        raise ModelFitError(f"implausible sun swing {swing:.4f} C per W/m2")
    return SunFit(swing, lag, sunny, float(np.sqrt(np.mean(r**2))), error)
