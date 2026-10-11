"""How fast the room warms (and cools) when the room setpoint is changed, measured.

The room response model says what one degree of setpoint should do to the room over a day.
This checks it against what the room actually did after the setpoint changes in the history:
for every 24 h window the change of the room temperature is regressed on the change of the
mean setpoint (warming and cooling separately), with the weather change, the outdoor
temperature and the room level as controls. The same regression is run on the room the model
itself would have had under the same setpoints, so that the two are compared like for like
(a setpoint changes in the middle of windows and the room keeps adjusting from earlier
changes, which lowers both coefficients alike). The ratio is the warm-up scale. Recent windows count more, so a change in the house (new floor,
new heat curve) shows up. The scale is only used when it differs from 1 by more than twice its
uncertainty, and never by more than a factor 1.5 either way.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt

from core.theria.exceptions import ModelFitError

FloatArray = npt.NDArray[np.float64]

WINDOW_HOURS = 24
HALF_LIFE_DAYS = 365.0
SETTLE_DAYS = 10  # the modelled room starts at its level; let that wear off
MIN_PRESENT_SHARE = 0.7  # of the hours of a window that must have data
MIN_WINDOWS = 200
MIN_STEP_WINDOWS = 30
STEP_C = 1.0  # a window with a setpoint change of at least this much is a step
MIN_SCALE, MAX_SCALE = 0.5, 1.5
SIGNIFICANCE = 2.0  # standard errors


@dataclass(frozen=True)
class WarmupFit:
    scale: float  # measured / modelled warming over 24 h
    se: float
    applied_scale: (
        float  # what the planner uses (1.0 unless the difference is significant)
    )
    response_c_per_c: float  # measured room warming over 24 h per degree of setpoint
    predicted_c_per_c: (
        float  # what the same estimator gives on the model's own simulated room
    )
    cooling_response_c_per_c: float  # the same for lowering the setpoint (per degree)
    cooling_scale: float  # measured / modelled cooling (informational: the planner does not use it)
    cooling_se: float
    windows: int
    step_windows: int


def _window_mean(values: FloatArray, start: int, stop: int) -> float:
    """Mean of a window, or NaN when too many hours are missing."""
    window = values[start:stop]
    present = np.isfinite(window)
    if present.sum() < MIN_PRESENT_SHARE * len(window):
        return float("nan")
    return float(np.mean(window[present]))


def fit_warmup(
    hours: list[datetime],
    room_c: FloatArray,
    t_out_c: FloatArray,
    rts_c: FloatArray,
    gain_c_per_c: float,
    rate_per_day: float,
    flow_per_rts: float,
    level_ref_c: float,
) -> WarmupFit:
    n = len(hours)
    reference = hours[-1]
    # the room the model would have had under the same setpoints (a deviation from its level)
    modelled = np.zeros(n)
    centred = np.nan_to_num(rts_c - np.nanmean(rts_c), nan=0.0)
    for k in range(1, n):
        modelled[k] = modelled[k - 1] + rate_per_day / 24 * (
            gain_c_per_c * flow_per_rts * centred[k - 1] - modelled[k - 1]
        )
    rows: list[list[float]] = []
    modelled_change: list[float] = []
    weights: list[float] = []
    for i in range(SETTLE_DAYS * 24, n - WINDOW_HOURS, WINDOW_HOURS):
        start, end, before = i, i + WINDOW_HOURS, i - WINDOW_HOURS
        room_start, room_end = room_c[start], room_c[end]
        d_rts = _window_mean(rts_c, start, end) - _window_mean(rts_c, before, start)
        d_out = _window_mean(t_out_c, start, end) - _window_mean(t_out_c, before, start)
        outdoor = _window_mean(t_out_c, start, end)
        row = [room_end - room_start, d_rts, d_out, room_start - level_ref_c, outdoor]
        if not np.isfinite(row).all():
            continue
        rows.append(row)
        modelled_change.append(float(modelled[end] - modelled[start]))
        age_days = (reference - hours[i]).total_seconds() / 86400
        weights.append(0.5 ** (age_days / HALF_LIFE_DAYS))
    if len(rows) < MIN_WINDOWS:
        raise ModelFitError(
            f"the warm-up speed needs {MIN_WINDOWS} days of history, found {len(rows)}"
        )
    data, w = np.array(rows), np.array(weights)
    y, d_rts = data[:, 0], data[:, 1]
    step_windows = int(np.sum(np.abs(d_rts) >= STEP_C))
    if step_windows < MIN_STEP_WINDOWS:
        raise ModelFitError(
            f"the warm-up speed needs {MIN_STEP_WINDOWS} days with a setpoint change of "
            f"{STEP_C} C or more, found {step_windows}"
        )
    design = np.column_stack(
        [
            np.ones(len(y)),
            np.maximum(d_rts, 0.0),  # raising the setpoint
            np.minimum(d_rts, 0.0),  # lowering it
            data[:, 2],
            data[:, 3],
            data[:, 4],
        ]
    )
    root = np.sqrt(w)
    weighted = design * root[:, None]
    coef, *_ = np.linalg.lstsq(weighted, y * root, rcond=None)
    simulated, *_ = np.linalg.lstsq(
        weighted, np.array(modelled_change) * root, rcond=None
    )
    residual = y - design @ coef
    dof = max(1.0, float(np.sum(w)) - design.shape[1])
    sigma2 = float(np.sum(w * residual**2) / dof)
    covariance = np.linalg.pinv(weighted.T @ weighted) * sigma2
    predicted = float(simulated[1])
    predicted_cooling = float(simulated[2])
    if predicted <= 0.0 or predicted_cooling <= 0.0:
        raise ModelFitError("the model implies no response to compare with")
    scale = float(coef[1]) / predicted
    se = float(np.sqrt(covariance[1, 1])) / predicted
    significant = abs(scale - 1.0) > SIGNIFICANCE * se
    applied = float(np.clip(scale, MIN_SCALE, MAX_SCALE)) if significant else 1.0
    return WarmupFit(
        scale=scale,
        se=se,
        applied_scale=applied,
        response_c_per_c=float(coef[1]),
        predicted_c_per_c=predicted,
        cooling_response_c_per_c=float(coef[2]),
        cooling_scale=float(coef[2]) / predicted_cooling,
        cooling_se=float(np.sqrt(covariance[2, 2])) / predicted_cooling,
        windows=len(rows),
        step_windows=step_windows,
    )
