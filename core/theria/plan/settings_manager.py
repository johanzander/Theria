"""The plan settings that can be changed in the UI, and how a change is checked, saved and applied.

The add-on options give the defaults. A change made in the UI is an override saved in a small file
next to them; it survives restarts and can be reset to the configured value. Entities and coordinates
stay in the add-on options.
"""

import json
import logging
import os
from dataclasses import dataclass, replace
from typing import Any

from core.theria.exceptions import ConfigurationError
from core.theria.plan.settings import PlanSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SettingInfo:
    group: str
    label: str
    description: str
    unit: str
    kind: str  # float | int | bool
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None


def _f(group: str, label: str, description: str, unit: str, lo: float, hi: float, step: float) -> SettingInfo:
    return SettingInfo(group, label, description, unit, "float", lo, hi, step)


def _i(group: str, label: str, description: str, unit: str, lo: int, hi: int) -> SettingInfo:
    return SettingInfo(group, label, description, unit, "int", lo, hi, 1)


EDITABLE: dict[str, SettingInfo] = {
    "floor_c": _f("Comfort", "Comfort floor (day)", "The room is never planned below this between 06:00 and 24:00.", "°C", 10, 25, 0.5),
    "target_c": _f("Comfort", "Comfort target (day)", "The temperature you want on average; being below it costs the comfort cost per degree-hour.", "°C", 10, 25, 0.5),
    "ceiling_c": _f("Comfort", "Room limit", "The warmest a room may be planned to get. The LK thermostats are held at this in active mode (not above the LK limit).", "°C", 15, 30, 0.5),
    "extreme_floor_c": _f("Comfort", "Floor at extreme prices", "A lower floor that is allowed in the dearest hours.", "°C", 10, 25, 0.5),
    "night_floor_c": _f("Comfort", "Comfort floor (night)", "The floor between the night's start and end.", "°C", 10, 25, 0.5),
    "night_target_c": _f("Comfort", "Comfort target (night)", "The target at night; nobody is in the ground floor then.", "°C", 10, 25, 0.5),
    "night_start_hour": _i("Comfort", "Night starts", "Local hour the night starts (equal start and end switches the night off).", "hour", 0, 23),
    "night_end_hour": _i("Comfort", "Night ends", "Local hour the night ends and comfort is due.", "hour", 0, 24),
    "rts_min_c": _f("Pump setpoint", "Lowest setpoint", "The plan never asks for a setpoint below this (the curve offset, not a room temperature).", "°C", 10, 25, 0.5),
    "rts_max_c": _f("Pump setpoint", "Highest setpoint", "The plan never asks for a setpoint above this.", "°C", 10, 25, 0.5),
    "rts_base_c": _f("Pump setpoint", "Base setpoint", "What the flat baseline holds and what is restored when control stops.", "°C", 10, 25, 0.5),
    "rts_step_c": _f("Pump setpoint", "Setpoint step", "The size of one setpoint step the plan chooses between.", "°C", 0.1, 1, 0.1),
    "max_rts_step_per_hour_c": _f("Pump setpoint", "Largest rise per hour", "How far the setpoint may be raised in one hour (lowering is not limited).", "°C", 0.5, 5, 0.5),
    "lk_limit_c": _f("Floor and valves", "LK thermostat setpoint (active)", "Where the LK thermostats are held while Theria controls them: the valves close when a room reaches it.", "°C", 15, 30, 0.5),
    "lk_base_c": _f("Floor and valves", "LK thermostat setpoint (otherwise)", "Where they stand when Theria does not control them (shadow, off, stop).", "°C", 15, 30, 0.5),
    "floor_mix_max_c": _f("Floor and valves", "Floor mixing valve limit", "The most flow temperature the floor loops receive, whatever the pump sends.", "°C", 20, 60, 1),
    "max_flow_c": _f("Floor and valves", "Largest flow temperature asked for", "A sanity bound on the flow temperature the plan may ask the pump for.", "°C", 30, 60, 1),
    "boost_flow_limit_c": _f("Floor and valves", "Flow where the electric heater starts", "The plan does not raise the flow above this unless the baseline already is above.", "°C", 30, 60, 1),
    "regulator_active": SettingInfo("Floor and valves", "Regulator closes valves", "On: the regulator closes a room's valves early; off: it only logs what it would do.", "", "bool"),
    "comfort_cost_sek_per_c_hour": _f("Weights", "Comfort cost", "What an hour one degree below the target costs. Lower: the plan accepts a cooler room to save money.", "kr per °C·h", 0, 20, 0.1),
    "setpoint_deviation_cost_sek_per_c_hour": _f("Weights", "Cost of moving the setpoint", "Stands in for the fast rooms (salon, guest house): every hour away from the base setpoint costs this.", "kr per °C·h", 0, 5, 0.05),
    "min_saving_sek": _f("Weights", "Minimum saving", "A plan that saves less than this over the horizon is replaced by the flat setpoint (0 switches it off).", "kr", 0, 50, 0.1),
    "extreme_price_percentile": _f("Prices", "Extreme price percentile", "Hours above this percentile of the recent prices (and the minimum below) count as extreme.", "%", 50, 99.9, 1),
    "extreme_price_min_sek_per_kwh": _f("Prices", "Extreme price minimum", "A price below this is never extreme.", "kr/kWh", 0, 20, 0.1),
    "extreme_history_days": _i("Prices", "Extreme price history", "How many days of prices the percentile is taken over.", "days", 1, 60),
}


def _coerce(key: str, info: SettingInfo, value: Any) -> Any:
    if info.kind == "bool":
        if not isinstance(value, bool):
            raise ConfigurationError(f"{key} must be true or false")
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigurationError(f"{key} must be a number")
    if info.kind == "int":
        if value != int(value):
            raise ConfigurationError(f"{key} must be a whole number")
        value = int(value)
    else:
        value = float(value)
    if info.minimum is not None and value < info.minimum:
        raise ConfigurationError(f"{key} must be at least {info.minimum:g}")
    if info.maximum is not None and value > info.maximum:
        raise ConfigurationError(f"{key} must be at most {info.maximum:g}")
    return value


class SettingsManager:
    def __init__(self, base: PlanSettings, path: str | None):
        self._base = base
        self._path = path
        self._overrides: dict[str, Any] = {}
        self.current = base
        self._load()

    def _load(self) -> None:
        saved: dict[str, Any] = {}
        if self._path and os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                logger.warning("cannot read the settings overrides %s", self._path)
        for key, value in saved.items():
            try:
                info = EDITABLE[key]
                self._overrides[key] = _coerce(key, info, value)
                self.current = replace(self._base, **self._overrides)
            except (KeyError, ConfigurationError):
                self._overrides.pop(key, None)
                logger.warning("ignoring the saved setting %s=%r", key, value)
        self.current = replace(self._base, **self._overrides)

    def catalog(self) -> list[dict[str, Any]]:
        out = []
        for key, info in EDITABLE.items():
            out.append(
                {
                    "key": key,
                    "group": info.group,
                    "label": info.label,
                    "description": info.description,
                    "unit": info.unit,
                    "kind": info.kind,
                    "minimum": info.minimum,
                    "maximum": info.maximum,
                    "step": info.step,
                    "value": getattr(self.current, key),
                    "default": getattr(self._base, key),
                    "source": "override" if key in self._overrides else "default",
                }
            )
        return out

    def update(self, values: dict[str, Any]) -> PlanSettings:
        """Check and save a change; raises ConfigurationError and changes nothing when it is not valid."""
        changes = {}
        for key, value in values.items():
            if key not in EDITABLE:
                raise ConfigurationError(f"{key} is not editable here")
            changes[key] = _coerce(key, EDITABLE[key], value)
        merged = {**self._overrides, **changes}
        candidate = replace(self._base, **merged)  # runs the settings' own checks
        self._overrides = merged
        self.current = candidate
        self._save()
        return candidate

    def reset(self, key: str) -> PlanSettings:
        if key not in EDITABLE:
            raise ConfigurationError(f"{key} is not editable here")
        self._overrides.pop(key, None)
        self.current = replace(self._base, **self._overrides)
        self._save()
        return self.current

    def _save(self) -> None:
        if not self._path:
            return
        temporary = f"{self._path}.tmp"
        with open(temporary, "w") as f:
            json.dump(self._overrides, f, indent=1)
        os.replace(temporary, self._path)
