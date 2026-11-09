"""Configuration for the price-driven heating plan."""

from dataclasses import dataclass, field, fields
from datetime import datetime
from zoneinfo import ZoneInfo

from core.theria.exceptions import ConfigurationError
from core.theria.options import load_options

# A plan is always made: shadow plans without writing, active plans and writes (off was removed 2026-11-08)
SUPPORTED_MODES = ("shadow", "active")
# Options Home Assistant may still have stored from earlier versions; they are read past
REMOVED_OPTIONS = ("radiator_entities",)
MAX_SHIFT_RTS_C = 2.0  # the heat shift never moves the setpoint further from its base


@dataclass(frozen=True)
class RoomControl:
    """One ground-floor room as the regulator sees it: its LK thermostat and its temperature sensors
    (several are averaged)."""

    room: str
    thermostat_entity: str
    temperature_entities: tuple[str, ...]


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
    room_controls: tuple[RoomControl, ...] = ()
    # False: the regulator measures, decides and logs what it would do, but does not close valves.
    regulator_active: bool = False
    # The plan may close the floor loops for an hour (the LK thermostats held at the hour's floor)
    # instead of only cutting the pump setpoint: the pump has no room sensor and makes heat for
    # whatever the open loops draw, so closed loops are the way to stop the draw. The loops reopen by
    # themselves if a room falls to the floor.
    plan_closes_valves: bool = True
    # What the pump still uses in an hour with the floor loops closed, as a share of its energy at the
    # lowest setpoint (hot water, radiators, standby). A prior: lower it as measured closed hours show.
    closed_energy_share: float = 0.15
    # cumulative compressor electricity for hot water (kWh), for the energy forecast
    hot_water_energy_entity: str = "sensor.compr_cons_hotwat"
    price_area: str = "SE4"
    mode: str = "shadow"
    floor_c: float = (
        18.5  # savings study: a floor of 18.5 C costs nothing against 17-18 C and keeps the room above it
    )
    extreme_floor_c: float = 17.0
    target_c: float = 19.5
    ceiling_c: float = 22.0  # the room limit U: the plan may store heat up to it
    # Comfort is wanted from 06:00 to 24:00 (local time); nobody is in the ground-floor rooms at
    # night, so the room may be cooler then. start == end switches the night window off.
    timezone: str = "Europe/Stockholm"
    night_start_hour: int = 0
    night_end_hour: int = 6
    night_target_c: float = 18.5
    night_floor_c: float = 17.5
    # The LK thermostats are a static limit per room, held at the room limit: a room's valves close
    # when it reaches it (zone-control design). Not a lever the plan moves.
    lk_limit_c: float = 22.0
    # Where the thermostats stand when Theria does not control them (kill switch, shadow, off): the
    # setting the heat curve was tuned for.
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
    comfort_cost_sek_per_c_hour: float = 1.0
    # What an hour one degree under the floor costs. Finite on purpose: a hard floor (100) made the
    # plan heat at any price whenever the forecast missed the floor, as on 2026-11-04 at 4 kr/kWh,
    # when raising 20 instead of 15 gained 0.02 C. At this value the price always counts.
    floor_violation_cost_sek_per_c_hour: float = 5.0
    # Stands in for the rooms that react faster than the ground floor (salon, guest house):
    # every degree-hour the setpoint is away from the base setpoint costs this much.
    setpoint_deviation_cost_sek_per_c_hour: float = 0.1
    # Savings study (documentation/savings-potential): with the floor loops held at 40 C, day-ahead
    # planning gains most from cutting deep (12 C: 8.5 %, 15 C: 6.9 %, 16 C: 5.6 %) and nothing above 22 C. The
    # house has run setpoints down to 13 C for only about 130 hours below 15.5 C, so the lower limit stays at 15 until
    # step tests show how the pump answers a deeper cut. The old manual control moved the setpoint only about +/- 2 C (16-20).
    # min_authority limits the range until the model proves itself.
    rts_min_c: float = 15.0
    rts_max_c: float = 22.0
    rts_base_c: float = 18.0
    rts_step_c: float = 0.5
    max_rts_step_per_hour_c: float = 2.0
    # The least share of the setpoint range (rts_min..rts_max around the base) the plan may use; it
    # grows to the whole range as the model proves it predicts the room (see skill.py).
    min_authority: float = 0.75
    # A plan must beat the flat setpoint by at least this much (SEK over the horizon, comfort
    # cost included); otherwise the flat setpoint is kept and nothing is moved for a few öre. 0 switches it off.
    min_saving_sek: float = 0.2
    # The first hour keeps what was applied last (the setpoint and whether the floor loops are
    # closed) unless another choice is better by more than this (SEK over the horizon). The plan is
    # recomputed every 15 minutes from noisy inputs; without this a near tie flips the pump.
    commit_margin_sek: float = 0.3
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
    # How the plan is made: "heat_shift" moves heat from dear to cheap windows, "optimizer" is the
    # dynamic programme.
    plan_strategy: str = "heat_shift"
    # The heat shift: how far the rooms may swing around where they are (+/-), how far the setpoint is
    # moved in a window, and the price difference a window must be worth.
    shift_room_swing_c: float = 0.5
    shift_rts_c: float = 1.0
    shift_min_gain_sek_per_kwh: float = 0.5

    @property
    def lk_setpoint_c(self) -> float:
        """Where the thermostats stand when nothing asks for more: the heat shift's normal target, or the limit."""
        return self.lk_base_c if self.plan_strategy == "heat_shift" else self.lk_limit_c

    @property
    def closed_share(self) -> float:
        """The pump energy left in an hour with the floor loops closed."""
        return (
            self.shift_closed_energy_share
            if self.plan_strategy == "heat_shift"
            else self.closed_energy_share
        )

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
        if not self.floor_c <= self.lk_base_c <= self.lk_limit_c:
            raise ConfigurationError(
                "the LK settings must satisfy floor <= lk_base_c <= lk_limit_c"
            )
        if self.ceiling_c > self.lk_limit_c + 1e-9:
            raise ConfigurationError(
                "the plan's ceiling must not be above lk_limit_c, where the valves close"
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
        if not 0 <= self.closed_energy_share <= 1:
            raise ConfigurationError("closed_energy_share must be between 0 and 1")
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
        if not 30.0 <= self.max_flow_c <= 62.0:
            raise ConfigurationError(
                "max_flow_c must be between 30 and 62 C (the pump limit)"
            )
        if not 30.0 <= self.boost_flow_limit_c <= self.max_flow_c:
            raise ConfigurationError(
                "boost_flow_limit_c must be between 30 C and max_flow_c"
            )
        if self.plan_strategy not in ("heat_shift", "optimizer"):
            raise ConfigurationError("plan_strategy must be heat_shift or optimizer")
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
        if self.setpoint_deviation_cost_sek_per_c_hour < 0:
            raise ConfigurationError(
                "setpoint_deviation_cost_sek_per_c_hour must not be negative"
            )
        if self.comfort_cost_sek_per_c_hour < 0:
            raise ConfigurationError("comfort_cost_sek_per_c_hour must not be negative")
        if self.floor_violation_cost_sek_per_c_hour < 0:
            raise ConfigurationError(
                "floor_violation_cost_sek_per_c_hour must not be negative"
            )
        if self.min_saving_sek < 0:
            raise ConfigurationError("min_saving_sek must not be negative")
        if self.commit_margin_sek < 0:
            raise ConfigurationError("commit_margin_sek must not be negative")
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
        if "room_controls" in converted:
            converted["room_controls"] = tuple(
                RoomControl(
                    item["room"],
                    item["thermostat_entity"],
                    tuple(item["temperature_entities"]),
                )
                for item in converted["room_controls"]
            )
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
