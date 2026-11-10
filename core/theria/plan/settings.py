"""Configuration for the price-driven heating plan."""

from dataclasses import dataclass, field, fields
from datetime import datetime
from zoneinfo import ZoneInfo

from core.theria.exceptions import ConfigurationError
from core.theria.options import load_options

# A plan is always made: shadow plans without writing, active plans and writes (off was removed 2026-11-08)
SUPPORTED_MODES = ("shadow", "active")
# Options Home Assistant may still have stored from earlier versions; they are read past. Most belonged
# to the dynamic-programming optimizer and the regulator, removed 2026-11-10.
REMOVED_OPTIONS = (
    "radiator_entities",
    "plan_strategy",
    "room_controls",
    "regulator_active",
    "closed_energy_share",
    "extreme_floor_c",
    "ceiling_c",
    "lk_limit_c",
    "comfort_cost_sek_per_c_hour",
    "floor_violation_cost_sek_per_c_hour",
    "setpoint_deviation_cost_sek_per_c_hour",
    "min_saving_sek",
    "commit_margin_sek",
    "extreme_price_percentile",
    "extreme_price_min_sek_per_kwh",
    "extreme_history_days",
    "max_flow_c",
    "boost_flow_limit_c",
)
MAX_SHIFT_RTS_C = 2.0  # the heat shift never moves the setpoint further from its base


@dataclass(frozen=True)
class PlanSettings:
    """Plan configuration. Defaults are the values agreed in the design spec."""

    # Home Assistant's built-in Nord Pool integration (nordpool.get_prices_for_date): the one source of prices,
    # for the plan, the charts and the measured days.
    price_config_entry: str
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
    # The plan may close the floor loops for an hour (the LK thermostats held at the hour's floor)
    # instead of only cutting the pump setpoint: the pump has no room sensor and makes heat for
    # whatever the open loops draw, so closed loops are the way to stop the draw. The loops reopen by
    # themselves if a room falls to the floor.
    plan_closes_valves: bool = True
    # cumulative compressor electricity for hot water (kWh), for the energy forecast
    hot_water_energy_entity: str = "sensor.compr_cons_hotwat"
    price_area: str = "SE4"
    mode: str = "shadow"
    floor_c: float = (
        18.5  # savings study: a floor of 18.5 C costs nothing against 17-18 C and keeps the room above it
    )
    target_c: float = 19.5
    # Comfort is wanted from 06:00 to 24:00 (local time); nobody is in the ground-floor rooms at
    # night, so the room may be cooler then. start == end switches the night window off.
    timezone: str = "Europe/Stockholm"
    night_start_hour: int = 0
    night_end_hour: int = 6
    night_target_c: float = 18.5
    night_floor_c: float = 17.5
    # The thermostats' normal target: where they stand outside a window and when Theria does not
    # control them (shadow, leaving active mode).
    lk_base_c: float = 19.5
    # In the heat shift: how far each room may fall in a sell, day and night. The thermostat stands
    # `lk_close_offset_c` above the limit, because a room reopens fully about that far below its target.
    lk_lower_day_c: float = 18.5
    lk_lower_night_c: float = 17.5
    lk_close_offset_c: float = 0.5
    # A buy just before a sell sets the thermostat here, so that the loops are open and the floor stores heat.
    lk_upper_c: float = 20.5
    lk_opening_margin_c: float = 0.5  # a room closer to the upper limit is only half open
    shift_buy_max_gap_h: int = 2  # a buy further from its sell is gone before the sell starts
    # What the pump still uses in a sell with the loops closed, as a share of its energy (measured about 0.6)
    shift_closed_energy_share: float = 0.6
    # What the mixing valves (shunts) allow the floor loops to receive, whatever the pump sends.
    floor_mix_max_c: float = 40.0
    # The range the pump setpoint is kept inside, and its base outside the windows.
    rts_min_c: float = 15.0
    rts_max_c: float = 22.0
    rts_base_c: float = 18.0
    rts_step_c: float = 0.5
    max_rts_step_per_hour_c: float = 2.0
    # The least share of the setpoint range (rts_min..rts_max around the base) the plan may use; it
    # grows to the whole range as the model proves it predicts the room (see skill.py).
    min_authority: float = 0.75
    # The heat shift: how far the rooms may swing around where they are (+/-), how far the setpoint is
    # moved in a window, and the price difference a window must be worth.
    shift_room_swing_c: float = 0.5
    shift_rts_c: float = 1.0
    shift_min_gain_sek_per_kwh: float = 0.5

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
        if not self.floor_c <= self.target_c:
            raise ConfigurationError("plan temperatures must satisfy floor <= target")
        if not (0 <= self.night_start_hour <= 23 and 0 <= self.night_end_hour <= 24):
            raise ConfigurationError("night hours must be between 0 and 24")
        if not (
            self.night_floor_c <= self.floor_c
            and self.night_floor_c <= self.night_target_c <= self.target_c
        ):
            raise ConfigurationError(
                "night temperatures must satisfy night_floor <= floor and "
                "night_floor <= night_target <= target"
            )
        if not self.lk_lower_night_c <= self.lk_lower_day_c <= self.lk_upper_c:
            raise ConfigurationError(
                "the LK limits must satisfy night lower <= day lower <= upper"
            )
        if (
            self.lk_close_offset_c < 0
            or self.lk_opening_margin_c < 0
            or self.shift_buy_max_gap_h < 0
        ):
            raise ConfigurationError(
                "the LK offset, opening margin and buy gap must not be negative"
            )
        if not 0 <= self.shift_closed_energy_share <= 1:
            raise ConfigurationError("shift_closed_energy_share must be between 0 and 1")
        if self.floor_mix_max_c <= 0:
            raise ConfigurationError("floor_mix_max_c must be positive")
        if not self.rts_min_c <= self.rts_base_c <= self.rts_max_c:
            raise ConfigurationError(
                "plan room setpoint range must satisfy rts_min <= rts_base <= rts_max"
            )
        if not 0 < self.min_authority <= 1:
            raise ConfigurationError("min_authority must be above 0 and at most 1")
        if self.rts_step_c <= 0 or self.max_rts_step_per_hour_c < self.rts_step_c:
            raise ConfigurationError(
                "rts_step_c must be positive and not larger than max_rts_step_per_hour_c"
            )
        if self.shift_rts_c > MAX_SHIFT_RTS_C:
            raise ConfigurationError(
                f"shift_rts_c must be at most {MAX_SHIFT_RTS_C}: larger steps have not been measured"
            )
        if self.shift_room_swing_c <= 0 or self.shift_rts_c <= 0:
            raise ConfigurationError(
                "shift_room_swing_c and shift_rts_c must be positive"
            )
        if self.shift_min_gain_sek_per_kwh < 0:
            raise ConfigurationError("shift_min_gain_sek_per_kwh must not be negative")
        if not self.room_temp_entities or not self.room_thermostat_entities:
            raise ConfigurationError(
                "room_temp_entities and room_thermostat_entities must not be empty"
            )

    @classmethod
    def from_dict(cls, data: dict) -> "PlanSettings":
        """Create from the `plan` section of config.yaml."""
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k not in REMOVED_OPTIONS}
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
