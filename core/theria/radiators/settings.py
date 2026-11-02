"""Configuration for the salon and guest house radiator control.

`RadiatorSettings` is what the add-on options fix (entities, physical constants); `RadiatorPrefs`
is what Johan edits in the UI (schedule, temperatures, bands) and the options only give defaults for.
"""

import math
from dataclasses import dataclass, field, fields
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from core.theria.exceptions import ConfigurationError
from core.theria.options import load_options
from core.theria.sensor_guard import GuardSettings

SUPPORTED_MODES = ("off", "shadow")
MIN_ROOM_C = 5.0  # Tado's range
MAX_ROOM_C = 25.0
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass(frozen=True)
class DaySchedule:
    open_at: time
    close_at: time

    def __post_init__(self) -> None:
        if not self.open_at < self.close_at:
            raise ConfigurationError("opening time must be before closing time")


@dataclass(frozen=True)
class GuestRoom:
    entity: str
    t_room_c: float
    band_c: float | None = None  # None: the global band

    def __post_init__(self) -> None:
        if not MIN_ROOM_C <= self.t_room_c <= MAX_ROOM_C:
            raise ConfigurationError(
                f"{self.entity}: the wanted temperature must be between {MIN_ROOM_C:g} and {MAX_ROOM_C:g} C"
            )
        if self.band_c is not None and not 0 < self.band_c <= 5:
            raise ConfigurationError("a band must be above 0 and at most 5 C")


def _default_schedule() -> dict[str, DaySchedule | None]:
    weekday = DaySchedule(time(9), time(19))
    weekend = DaySchedule(time(10), time(16))
    return {day: weekday if day in WEEKDAYS[:5] else weekend for day in WEEKDAYS}


def _parse_time(text: str) -> time:
    try:
        return time.fromisoformat(text)
    except ValueError as e:
        raise ConfigurationError(f"invalid time {text!r}") from e


@dataclass(frozen=True)
class RadiatorPrefs:
    """The values editable in the UI."""

    salon_schedule: dict[str, DaySchedule | None] = field(
        default_factory=_default_schedule
    )
    salon_open_target_c: float = 19.0  # mean of all points while open
    salon_closed_floor_c: float = 17.0  # the mean may cool this far when closed
    salon_preheat_c: float = 20.0
    salon_margin_c: float = 0.3  # above the target in the prediction
    salon_max_setpoint_c: float = 22.0
    guest_band_c: float = 1.5
    guest_rooms: tuple[GuestRoom, ...] = ()
    mode: str = "shadow"

    def __post_init__(self) -> None:
        numbers = (
            self.salon_open_target_c,
            self.salon_closed_floor_c,
            self.salon_preheat_c,
            self.salon_margin_c,
            self.salon_max_setpoint_c,
            self.guest_band_c,
        )
        if not all(math.isfinite(n) for n in numbers):
            raise ConfigurationError("radiator temperatures must be finite numbers")
        if set(self.salon_schedule) != set(WEEKDAYS):
            raise ConfigurationError("the schedule needs an entry for every weekday")
        if self.mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"radiator mode must be one of {SUPPORTED_MODES}, got {self.mode!r}"
            )
        if self.salon_closed_floor_c > self.salon_open_target_c:
            raise ConfigurationError("the closed floor must not exceed the open target")
        if not (
            self.salon_open_target_c + self.salon_margin_c
            <= self.salon_preheat_c
            <= self.salon_max_setpoint_c
        ):
            raise ConfigurationError(
                "salon temperatures must satisfy open_target + margin <= preheat <= max_setpoint"
            )
        if self.salon_margin_c < 0:
            raise ConfigurationError("salon_margin_c must not be negative")
        if not 0 < self.guest_band_c <= 5:
            raise ConfigurationError("a band must be above 0 and at most 5 C")

    def is_open(self, moment: datetime, tz: str) -> bool:
        local = moment.astimezone(ZoneInfo(tz))
        day = self.salon_schedule[WEEKDAYS[local.weekday()]]
        return day is not None and day.open_at <= local.time() < day.close_at

    def open_in_hour(self, hour_start: datetime, tz: str) -> bool:
        """Whether the shop is open at any time in the hour that starts at `hour_start`."""
        local = hour_start.astimezone(ZoneInfo(tz))
        day = self.salon_schedule[WEEKDAYS[local.weekday()]]
        if day is None:
            return False
        start = local.hour * 60 + local.minute
        opens = day.open_at.hour * 60 + day.open_at.minute
        closes = day.close_at.hour * 60 + day.close_at.minute
        return opens < start + 60 and closes > start

    def to_dict(self) -> dict[str, Any]:
        return {
            "salon_schedule": {
                day: (
                    None
                    if s is None
                    else {
                        "open": s.open_at.strftime("%H:%M"),
                        "close": s.close_at.strftime("%H:%M"),
                    }
                )
                for day, s in self.salon_schedule.items()
            },
            "salon_open_target_c": self.salon_open_target_c,
            "salon_closed_floor_c": self.salon_closed_floor_c,
            "salon_preheat_c": self.salon_preheat_c,
            "salon_margin_c": self.salon_margin_c,
            "salon_max_setpoint_c": self.salon_max_setpoint_c,
            "guest_band_c": self.guest_band_c,
            "guest_rooms": [
                {"entity": r.entity, "t_room_c": r.t_room_c, "band_c": r.band_c}
                for r in self.guest_rooms
            ],
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RadiatorPrefs":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown radiator preferences: {sorted(unknown)}")
        converted = dict(data)
        try:
            if "salon_schedule" in converted:
                converted["salon_schedule"] = {
                    day: (
                        None
                        if s is None
                        else DaySchedule(
                            _parse_time(s["open"]), _parse_time(s["close"])
                        )
                    )
                    for day, s in converted["salon_schedule"].items()
                }
            if "guest_rooms" in converted:
                converted["guest_rooms"] = tuple(
                    GuestRoom(r["entity"], float(r["t_room_c"]), r.get("band_c"))
                    for r in converted["guest_rooms"]
                )
            return cls(**converted)
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            raise ConfigurationError(f"invalid radiator preferences: {e!r}") from e


@dataclass(frozen=True)
class RadiatorSettings:
    """What the add-on options fix."""

    salon_thermostats: tuple[str, ...]
    salon_sensors: tuple[str, ...]  # the salon's other points (floor, ceiling)
    salon_sensor_guard: GuardSettings
    outdoor_entity: str
    defaults: RadiatorPrefs
    timezone: str = "Europe/Stockholm"
    salon_c_j_per_k: float = 2.33e6
    salon_u_w_per_k: float = 51.7
    salon_heat_power_w: float = 1800.0  # the highest power seen (1830 W); an estimate to check against the shadow log
    salon_baseline_c: float = (
        20.0  # the mean the thermostats at 21 C give today (estimate)
    )
    guest_loss_w_per_k: float = 100.0  # crude, for the indicative saving only

    def __post_init__(self) -> None:
        if not self.salon_thermostats:
            raise ConfigurationError("salon_thermostats must not be empty")
        if (
            min(self.salon_c_j_per_k, self.salon_u_w_per_k, self.salon_heat_power_w)
            <= 0
        ):
            raise ConfigurationError("salon model constants must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RadiatorSettings":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown radiator settings: {sorted(unknown)}")
        converted = dict(data)
        for key in ("salon_thermostats", "salon_sensors"):
            if key in converted:
                converted[key] = tuple(converted[key])
        try:
            converted["salon_sensor_guard"] = GuardSettings.from_dict(
                converted["salon_sensor_guard"]
            )
            converted["defaults"] = RadiatorPrefs.from_dict(converted["defaults"])
            return cls(**converted)
        except (KeyError, TypeError) as e:
            raise ConfigurationError(f"invalid radiator settings: {e!r}") from e

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "RadiatorSettings | None":
        section = options.get("radiators")
        return None if section is None else cls.from_dict(section)


def load_radiator_settings(
    options_path: str, config_path: str
) -> RadiatorSettings | None:
    options = load_options(options_path, config_path)
    return None if options is None else RadiatorSettings.from_options(options)
