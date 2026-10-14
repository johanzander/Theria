"""Bounded steps for the learned parameters: a refit may move them only so far per day.

The refit fits the whole history again, so one odd day (or a stretch of days with a stove, a
guest or a broken sensor) could move the night model or the warm-up scale a long way at once.
Each parameter therefore moves at most a fixed step from what the planner used until now
towards the new fit, and every cut is reported so that it shows in the parameter history.
The measurements themselves (scale, errors) are kept as fitted; only what the planner uses is
limited.
"""

from dataclasses import replace

from core.theria.plan.calibration import RoomFit
from core.theria.plan.daily_room import room_params_from_fit
from core.theria.plan.night_drift import NightDrift
from core.theria.plan.warmup import WarmupFit

MAX_INTERCEPT_STEP_C = 0.10  # night change at 0 C outdoors, per day
MAX_SLOPE_STEP = 0.02  # night change per degree outdoors, per day
MAX_LEVEL_STEP = 0.05  # night change per degree of room level, per day
MAX_APPLIED_SCALE_STEP = 0.10  # warm-up scale the planner uses, per day
MAX_GAIN_STEP = 0.05  # room response, degrees per degree of flow, per day
MAX_RATE_STEP_PER_DAY = 0.03  # share of the gap the room closes per day, per day
MAX_SWING_STEP = 0.001  # the sun's swing on the room, C per W/m2, per day


def limit_step(old: float, new: float, max_step: float) -> float:
    return old + max(-max_step, min(max_step, new - old))


def trim_night(
    old: NightDrift | None, new: NightDrift | None
) -> tuple[NightDrift | None, list[str]]:
    if old is None or new is None:
        return new, []
    notes: list[str] = []
    fields = {}
    for name, label, step in (
        ("intercept_c", "night intercept", MAX_INTERCEPT_STEP_C),
        ("slope_c_per_c", "night slope", MAX_SLOPE_STEP),
        ("level_c_per_c", "night level", MAX_LEVEL_STEP),
    ):
        before, fitted = getattr(old, name), getattr(new, name)
        limited = limit_step(before, fitted, step)
        if limited != fitted:
            notes.append(f"{label} {fitted:.3f} limited to {limited:.3f}")
            fields[name] = limited
    return replace(new, **fields), notes


def trim_warmup(
    old: WarmupFit | None, new: WarmupFit | None
) -> tuple[WarmupFit | None, list[str]]:
    if old is None or new is None:
        return new, []
    limited = limit_step(old.applied_scale, new.applied_scale, MAX_APPLIED_SCALE_STEP)
    if limited == new.applied_scale:
        return new, []
    note = f"warm-up scale {new.applied_scale:.2f} limited to {limited:.2f}"
    return replace(new, applied_scale=limited), [note]


def trim_room(
    old: RoomFit | None, new: RoomFit | None
) -> tuple[RoomFit | None, list[str]]:
    """The hourly parameters are rebuilt from the limited response, so they stay consistent."""
    if old is None or new is None:
        return new, []
    gain = limit_step(
        old.response.gain_c_per_c, new.response.gain_c_per_c, MAX_GAIN_STEP
    )
    rate = limit_step(
        old.response.rate_per_day, new.response.rate_per_day, MAX_RATE_STEP_PER_DAY
    )
    swing = limit_step(
        old.params.sun_swing_c_per_wm2, new.params.sun_swing_c_per_wm2, MAX_SWING_STEP
    )
    notes = []
    if gain != new.response.gain_c_per_c:
        notes.append(f"room gain {new.response.gain_c_per_c:.3f} limited to {gain:.3f}")
    if rate != new.response.rate_per_day:
        notes.append(f"room rate {new.response.rate_per_day:.3f} limited to {rate:.3f}")
    if swing != new.params.sun_swing_c_per_wm2:
        notes.append(
            f"sun swing {new.params.sun_swing_c_per_wm2:.4f} limited to {swing:.4f}"
        )
    if not notes:
        return new, []
    response = replace(
        new.response,
        gain_c_per_c=gain,
        rate_per_day=rate,
        time_constant_days=1.0 / rate,
    )
    params = replace(
        room_params_from_fit(response),
        sun_swing_c_per_wm2=swing,
        sun_lag_h=new.params.sun_lag_h,
    )
    return replace(new, params=params, response=response), notes
