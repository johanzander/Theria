"""Configuration for the price-driven heating plan."""

from dataclasses import dataclass, field, fields
from datetime import datetime
from zoneinfo import ZoneInfo

from core.theria.exceptions import ConfigurationError
from core.theria.options import load_options

SUPPORTED_MODES = ("off", "shadow", "active")


@dataclass(frozen=True)
class PlanSettings:
    """Plan configuration. Defaults are the values agreed in the design spec."""

    price_entity: str
    outdoor_entity: str
    room_setpoint_entity: str
    flow_setpoint_entity: str
    room_temp_entities: tuple[str, ...]
    room_thermostat_entities: tuple[str, ...]  # LK climate entities (setpoint history)
    room_history_entities: tuple[str, ...]  # older sensors that extend the room history
    heating_mode_entity: str  # on = hot water, off = heating
    flow_measured_entity: str
    energy_counter_entity: str  # cumulative heating electricity (kWh)
    aux_energy_entity: str  # cumulative electric heater energy in heating (kWh)
    aux_status_entity: str  # electric heater status (above 0 = running)
    latitude: float
    longitude: float
    # The old automation that wrote the room setpoint; the plan must not run alongside it.
    conflicting_automation_entity: str = "automation.set_room_temp_setpoint"
    # LK floor-loop actuators (binary sensors, on = open) per room name (config: a list of {room, entities})
    room_actuators: dict[str, tuple[str, ...]] = field(default_factory=dict)
    mode: str = "off"
    floor_c: float = 18.0
    extreme_floor_c: float = 17.0
    target_c: float = 19.5
    ceiling_c: float = 21.0
    # Comfort is wanted from 06:00 to 24:00 (local time); nobody is in the ground-floor rooms at
    # night, so the room may be cooler then. start == end switches the night window off.
    timezone: str = "Europe/Stockholm"
    night_start_hour: int = 0
    night_end_hour: int = 6
    night_target_c: float = 18.5
    night_floor_c: float = 17.5
    lk_setpoint_c: float = 21.0  # LK thermostat setpoint while pre-heating (the boost)
    lk_base_c: float = (
        19.5  # LK thermostat setpoint otherwise (never lowered below this)
    )
    comfort_cost_sek_per_c_hour: float = 1.0
    # Stands in for the rooms that react faster than the ground floor (salon, guest house):
    # every degree-hour the setpoint is away from the base setpoint costs this much.
    setpoint_deviation_cost_sek_per_c_hour: float = 0.1
    rts_min_c: float = 16.0  # the old manual control moved the setpoint about +/- 2 C
    rts_max_c: float = 20.0
    rts_base_c: float = 18.0
    rts_step_c: float = 0.5
    max_rts_step_per_hour_c: float = 1.0
    extreme_price_percentile: float = 95.0
    extreme_price_min_sek_per_kwh: float = 3.0
    extreme_history_days: int = 14
    # Highest pump flow temperature the plan may ask for: the 45 C flow temperature LK
    # designed the floor for (LK023393). Above it the floor is outside its design and the
    # parquet surface leaves the 25 C range LK calculated.
    max_flow_c: float = 45.0
    # Above this flow target the heat pump's electric heater starts to help (in the 2022-2026
    # history: no use below 40 C, 1.6 % of hours at 40-44 C, 21 % of hours above 44 C, and mostly
    # when it is cold). The plan does not boost above it unless the baseline already is above.
    boost_flow_limit_c: float = 42.0

    def in_night(self, moment: datetime) -> bool:
        """Whether the local time of `moment` is inside the night window (comfort not needed)."""
        if self.night_start_hour == self.night_end_hour:
            return False
        hour = moment.astimezone(ZoneInfo(self.timezone)).hour
        if self.night_start_hour < self.night_end_hour:
            return self.night_start_hour <= hour < self.night_end_hour
        return hour >= self.night_start_hour or hour < self.night_end_hour

    def comfort_starts_at(self, moment: datetime) -> bool:
        """Whether `moment` falls in the first hour of comfort (the hour the night ends)."""
        if self.night_start_hour == self.night_end_hour:
            return False
        local = moment.astimezone(ZoneInfo(self.timezone))
        return local.hour == self.night_end_hour % 24

    def __post_init__(self) -> None:
        if self.mode == "active":
            raise ConfigurationError(
                "active mode cannot be set in the configuration; switch to it through "
                "the API with confirm=true"
            )
        if self.mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"plan mode must be one of {SUPPORTED_MODES}, got {self.mode!r}"
            )
        if not self.extreme_floor_c <= self.floor_c <= self.target_c <= self.ceiling_c:
            raise ConfigurationError(
                "plan temperatures must satisfy extreme_floor <= floor <= target <= ceiling"
            )
        if not (0 <= self.night_start_hour <= 23 and 0 <= self.night_end_hour <= 24):
            raise ConfigurationError("night hours must be between 0 and 24")
        if not (
            self.extreme_floor_c <= self.night_floor_c <= self.floor_c
            and self.night_floor_c <= self.night_target_c <= self.target_c
        ):
            raise ConfigurationError(
                "night temperatures must satisfy extreme_floor <= night_floor <= floor and "
                "night_floor <= night_target <= target"
            )
        if not self.floor_c <= self.lk_base_c <= self.lk_setpoint_c:
            raise ConfigurationError(
                "LK setpoints must satisfy floor <= lk_base_c <= lk_setpoint_c"
            )
        if not self.rts_min_c <= self.rts_base_c <= self.rts_max_c:
            raise ConfigurationError(
                "plan room setpoint range must satisfy rts_min <= rts_base <= rts_max"
            )
        if self.rts_step_c <= 0 or self.max_rts_step_per_hour_c < self.rts_step_c:
            raise ConfigurationError(
                "rts_step_c must be positive and not larger than max_rts_step_per_hour_c"
            )
        if not 30.0 <= self.max_flow_c <= 62.0:
            raise ConfigurationError(
                "max_flow_c must be between 30 and 62 C (the pump limit)"
            )
        if not 30.0 <= self.boost_flow_limit_c <= self.max_flow_c:
            raise ConfigurationError(
                "boost_flow_limit_c must be between 30 C and max_flow_c"
            )
        if self.setpoint_deviation_cost_sek_per_c_hour < 0:
            raise ConfigurationError(
                "setpoint_deviation_cost_sek_per_c_hour must not be negative"
            )
        if self.comfort_cost_sek_per_c_hour < 0:
            raise ConfigurationError("comfort_cost_sek_per_c_hour must not be negative")
        if not 0 < self.extreme_price_percentile < 100:
            raise ConfigurationError("extreme_price_percentile must be in (0, 100)")
        if not self.room_temp_entities or not self.room_thermostat_entities:
            raise ConfigurationError(
                "room_temp_entities and room_thermostat_entities must not be empty"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "PlanSettings":
        """Create from the `plan` section of config.yaml."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(f"unknown plan settings: {sorted(unknown)}")
        converted = dict(data)
        for key in (
            "room_temp_entities",
            "room_thermostat_entities",
            "room_history_entities",
        ):
            if key in converted:
                converted[key] = tuple(converted[key])
        if "room_actuators" in converted:
            converted["room_actuators"] = {
                item["room"]: tuple(item["entities"])
                for item in converted["room_actuators"]
            }
        try:
            return cls(**converted)
        except TypeError as e:
            raise ConfigurationError(f"invalid plan settings: {e}") from e

    @classmethod
    def from_options(cls, options: dict) -> "PlanSettings | None":
        """Settings from the add-on options, or None when no `plan` section exists."""
        section = options.get("plan")
        return None if section is None else cls.from_dict(section)


def load_plan_settings(options_path: str, config_path: str) -> PlanSettings | None:
    """Load the `plan` section: add-on options.json in production, config.yaml in development.

    Returns None when neither file exists or when there is no `plan` section.
    """
    options = load_options(options_path, config_path)
    return None if options is None else PlanSettings.from_options(options)
