"""How much the room cools during the night, as a function of the outdoor temperature.

The room response model (`daily_room`) says how the flow temperature moves the room over days.
It cannot say that on a cold night the room simply loses heat: the plan anchors the drift so
that a flat setpoint holds the room, which hides it. This model learns the missing part from
the nights themselves: for every night (the window in which comfort is not needed, so no
stove, people or sun) the change of the room temperature from the start to the end of the
night is regressed on the mean outdoor temperature. Only nights with a steady setpoint count,
and recent nights count more, so the model follows the house as it (and the heat curve) changes.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

from core.theria.exceptions import ModelFitError
from core.theria.plan.pump_model import DEFAULT_FLOW_PER_RTS
from core.theria.plan.room_model import PRIOR_GAIN_C_PER_C, PRIOR_RATE_PER_DAY

FloatArray = npt.NDArray[np.float64]

HALF_LIFE_DAYS = 150.0  # a night this much older counts half as much
MIN_NIGHTS = 25
MIN_OUTDOOR_SPREAD_C = 6.0  # p90 - p10 of the nights' mean outdoor temperature
MAX_SETPOINT_RANGE_C = (
    2.0  # most the setpoint may have moved (what the widest stage allows)
)
LEAD_IN_HOURS = 6
# What one degree of setpoint held for an hour does to the room, with the valves open: the room
# gain and rate of the prior response times the flow temperature per degree of setpoint. Small
# (hundredths of a degree over a night), but it lets the nights the plan moved the setpoint in
# teach the model too.
SETPOINT_EFFECT_C_PER_C_H = (
    PRIOR_GAIN_C_PER_C * PRIOR_RATE_PER_DAY / 24 * DEFAULT_FLOW_PER_RTS
)
RECENT_NIGHTS = 14
MAX_COOLING_C_PER_H = 0.4  # sanity limit on what the model may hand to the planner


@dataclass(frozen=True)
class NightObservation:
    day: date
    t_out_c: float
    measured_change_c: float
    predicted_change_c: float


@dataclass(frozen=True)
class NightDrift:
    """Room change over the night =
    intercept_c + slope_c_per_c * mean outdoor temperature + level_c_per_c * (room - level_ref_c)

    A warmer room loses more (level_c_per_c < 0). The level has to be in the model: the cold
    nights were mostly nights with a cool room, and without it the effect of the cold is hidden.
    """

    intercept_c: float
    slope_c_per_c: (
        float  # change per degree WARMER outdoors (> 0: colder = more cooling)
    )
    level_c_per_c: float  # change per degree the room starts the night warmer (<= 0)
    level_ref_c: float  # the room temperature the intercept refers to
    night_hours: int
    nights: int
    rmse_c: float
    recent: tuple[NightObservation, ...]


def night_change_c(drift: NightDrift, t_out_c: float, room_c: float) -> float:
    """Expected room temperature change over a whole night."""
    return (
        drift.intercept_c
        + drift.slope_c_per_c * t_out_c
        + drift.level_c_per_c * (room_c - drift.level_ref_c)
    )


def drift_per_hour(drift: NightDrift, t_out_c: float, room_c: float) -> float:
    """The same, per night hour, limited to what is physically reasonable."""
    per_hour = night_change_c(drift, t_out_c, room_c) / drift.night_hours
    return float(np.clip(per_hour, -MAX_COOLING_C_PER_H, MAX_COOLING_C_PER_H))


def fit_night_drift(
    hours: list[datetime],
    room_c: FloatArray,
    t_out_c: FloatArray,
    rts_c: FloatArray,
    start_hour: int,
    end_hour: int,
    timezone: str,
    level_ref_c: float,
) -> NightDrift:
    """Fit the night cooling from hourly history (hours in UTC, all arrays aligned)."""
    length = (end_hour - start_hour) % 24
    if length == 0:
        raise ModelFitError("there is no night window to learn the night cooling from")
    zone = ZoneInfo(timezone)
    reference = hours[-1]
    days: list[date] = []
    temps: list[float] = []
    levels: list[float] = []
    changes: list[float] = []
    weights: list[float] = []
    for i, moment in enumerate(hours):
        local = moment.astimezone(zone)
        if local.hour != start_hour or i + length >= len(hours) or i < LEAD_IN_HOURS:
            continue
        window = slice(i, i + length)
        room_start, room_end = room_c[i], room_c[i + length]
        outdoor = (
            np.nanmean(t_out_c[window])
            if np.isfinite(t_out_c[window]).any()
            else np.nan
        )
        setpoints = rts_c[i - LEAD_IN_HOURS : i + length + 1]
        if not np.isfinite([room_start, room_end, outdoor]).all():
            continue
        if not np.isfinite(setpoints).all() or np.ptp(setpoints) > MAX_SETPOINT_RANGE_C:
            continue
        age_days = (reference - moment).total_seconds() / 86400
        moved = float(np.sum(rts_c[window] - rts_c[i - 1]))
        days.append(local.date())
        temps.append(float(outdoor))
        levels.append(float(room_start))
        changes.append(float(room_end - room_start) - SETPOINT_EFFECT_C_PER_C_H * moved)
        weights.append(0.5 ** (age_days / HALF_LIFE_DAYS))
    if len(days) < MIN_NIGHTS:
        raise ModelFitError(
            f"the night cooling needs {MIN_NIGHTS} nights with a steady setpoint, found {len(days)}"
        )
    t, y, w = np.array(temps), np.array(changes), np.array(weights)
    level = np.array(levels) - level_ref_c
    if np.percentile(t, 90) - np.percentile(t, 10) < MIN_OUTDOOR_SPREAD_C:
        raise ModelFitError("the nights are too alike in outdoor temperature to fit")
    root = np.sqrt(w)

    def regress(columns: list[FloatArray]) -> FloatArray:
        design = np.column_stack(columns)
        coef, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        return np.asarray(coef, dtype=float)

    ones = np.ones(len(t))
    coef = regress([ones, t, level])
    if coef[2] > 0.0:
        # a warmer room that cools LESS is not physical: leave the level out rather than trust it
        two = regress([ones, t])
        coef = np.array([two[0], two[1], 0.0])
    intercept, slope, level_slope = float(coef[0]), float(coef[1]), float(coef[2])
    if slope <= 0.0:
        raise ModelFitError(
            f"colder nights do not cool the room more (slope {slope:.3f} C per C)"
        )
    predicted = intercept + slope * t + level_slope * level
    rmse = float(np.sqrt(np.average((predicted - y) ** 2, weights=w)))
    recent = tuple(
        NightObservation(days[k], temps[k], changes[k], float(predicted[k]))
        for k in range(len(days) - RECENT_NIGHTS, len(days))
    )
    return NightDrift(
        intercept, slope, level_slope, level_ref_c, length, len(days), rmse, recent
    )


def drift_pattern(
    drift: NightDrift,
    in_night: Callable[[datetime], bool],
    hour_starts: list[datetime],
    t_out_c: FloatArray,
    room_c: float,
) -> FloatArray:
    """Room temperature change per hour from what the room does on its own, for the planner.

    The night hours get the learned cooling at that hour's outdoor temperature. The same amount
    is given back over the other hours, so that over the horizon the pattern adds up to zero:
    the planner's anchoring (a flat setpoint holds the room on average) stays true and only the
    shape of the day changes. Without a night hour in the horizon there is no pattern.
    """
    night = np.array([in_night(start) for start in hour_starts], dtype=bool)
    pattern = np.zeros(len(hour_starts))
    if not night.any() or night.all():
        return pattern
    pattern[night] = [drift_per_hour(drift, float(t), room_c) for t in t_out_c[night]]
    pattern[~night] = -float(np.sum(pattern[night])) / int((~night).sum())
    return pattern
