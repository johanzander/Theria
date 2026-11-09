"""The rules around the heat shift: heat is moved and not cut, a started window runs to its end, and
measured readings end it when the house cannot take it.

A window that a guard has ended stays ended until its end, so that a reading around a limit cannot
switch it on and off.
"""

import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from core.theria.plan.settings import PlanSettings

BALANCE_TOLERANCE = 0.1  # the share of the moved hours that may stay unmatched
THERMOSTAT_STEP_C = 0.5
CLOSING = "closing"  # the loops are closed by the thermostat
OPENING = (
    "opening"  # the thermostat stands above the room so that the floor stores heat
)
NORMAL = "normal"


@dataclass(frozen=True)
class SalonStatus:
    mean_c: float | None  # None: too few valid points to know
    floor_c: float


@dataclass(frozen=True)
class ShiftWindow:
    end: datetime
    adjust_c: float
    ended_by: str | None = None  # the guard that ended it

    def to_dict(self) -> dict[str, Any]:
        return {
            "end": self.end.isoformat(),
            "adjust_c": self.adjust_c,
            "ended_by": self.ended_by,
        }

    @staticmethod
    def from_dict(body: dict[str, Any]) -> "ShiftWindow":
        return ShiftWindow(
            datetime.fromisoformat(body["end"]),
            float(body["adjust_c"]),
            body["ended_by"],
        )


@dataclass(frozen=True)
class ShiftReadings:
    """What is measured now, and the window that has started."""

    room_c: float
    salon: SalonStatus | None = None  # None: the house has no salon to look after
    window: ShiftWindow | None = None
    # each LK room against the mean of them: a mean hides a cold room next to a warm one
    room_offsets_c: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "room_c": self.room_c,
            "salon": (
                None
                if self.salon is None
                else {"mean_c": self.salon.mean_c, "floor_c": self.salon.floor_c}
            ),
            "window": None if self.window is None else self.window.to_dict(),
            "room_offsets_c": list(self.room_offsets_c),
        }

    @staticmethod
    def from_dict(body: dict[str, Any]) -> "ShiftReadings":
        return ShiftReadings(
            float(body["room_c"]),
            None if body["salon"] is None else SalonStatus(**body["salon"]),
            None if body["window"] is None else ShiftWindow.from_dict(body["window"]),
            tuple(body.get("room_offsets_c", ())),  # absent in older runs
        )


def balanced(adjust: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """The adjustments with the least valuable unmatched hours given up, so that heat is moved, not cut."""
    out = adjust.copy()
    while abs(out.sum()) > BALANCE_TOLERANCE * np.abs(out).sum():
        if out.sum() < 0:
            sells = np.flatnonzero(out < 0)
            out[sells[np.argmin(prices[sells])]] = 0.0
        else:
            buys = np.flatnonzero(out > 0)
            out[buys[np.argmax(prices[buys])]] = 0.0
    return out


def guard_reason(adjust_c: float, shift: ShiftReadings) -> str | None:
    """Why this adjustment must not go on, or None. Each LK room is guarded by its own thermostat."""
    if adjust_c > 0:
        return None
    if shift.salon is None:
        return None
    if shift.salon.mean_c is None:
        return "salon_unknown"
    return "salon_floor" if shift.salon.mean_c < shift.salon.floor_c else None


def _fill(out: np.ndarray, hour_starts: list[datetime], window: ShiftWindow) -> None:
    value = 0.0 if window.ended_by else window.adjust_c
    for t, start in enumerate(hour_starts):
        if start >= window.end:
            break
        out[t] = value


def _run_end(adjust: np.ndarray, hour_starts: list[datetime]) -> datetime:
    """When the run of equal adjustments that starts at the first hour ends."""
    changes = np.flatnonzero(adjust != adjust[0])
    if len(changes):
        return hour_starts[int(changes[0])]
    return hour_starts[-1] + timedelta(hours=1)


def hold_window(
    adjust: np.ndarray,
    hour_starts: list[datetime],
    shift: ShiftReadings,
) -> tuple[np.ndarray, ShiftWindow | None]:
    """The adjustments with the started window kept or ended, and that window."""
    out = adjust.copy()
    window = (
        shift.window
        if shift.window is not None and shift.window.end > hour_starts[0]
        else None
    )
    if window is not None:
        _fill(out, hour_starts, window)
    elif out[0] != 0.0:
        window = ShiftWindow(_run_end(out, hour_starts), float(out[0]))
    if window is None or window.ended_by is not None:
        return out, window
    reason = guard_reason(window.adjust_c, shift)
    if reason is None:
        return out, window
    window = replace(window, ended_by=reason)
    _fill(out, hour_starts, window)
    return out, window


def thermostat_modes(
    adjust: np.ndarray,
    hour_starts: list[datetime],
    window: ShiftWindow | None,
    buy_max_gap_h: int,
) -> list[str]:
    """What the thermostats do each hour. Closing goes before opening.

    A sell closes, also after a guard has ended it for the pump: a room near its limit is not opened at
    the dearest price. A buy opens only if it ends at most `buy_max_gap_h` before a sell starts.
    """
    selling = np.array(adjust < 0)
    if window is not None and window.adjust_c < 0:
        selling |= np.array([start < window.end for start in hour_starts])
    modes = [CLOSING if sold else NORMAL for sold in selling]
    n = len(adjust)
    i = 0
    while i < n:
        if adjust[i] <= 0 or selling[i]:
            i += 1
            continue
        end = i
        while end < n and adjust[end] > 0 and not selling[end]:
            end += 1
        following = np.flatnonzero(selling[end:])
        if len(following) and following[0] <= buy_max_gap_h:
            modes[i:end] = [OPENING] * (end - i)
        i = end
    return modes


def thermostat_target(
    mode: str, night: bool, room_c: float | None, settings: PlanSettings
) -> float:
    """The thermostat's target for one room this hour."""
    if mode == CLOSING:
        lower = settings.lk_lower_night_c if night else settings.lk_lower_day_c
        value = lower + settings.lk_close_offset_c
        return math.ceil(value / THERMOSTAT_STEP_C - 1e-9) * THERMOSTAT_STEP_C
    if (
        mode == OPENING
        and room_c is not None
        and room_c < settings.lk_upper_c - settings.lk_opening_margin_c
    ):
        return settings.lk_upper_c
    return settings.lk_base_c
