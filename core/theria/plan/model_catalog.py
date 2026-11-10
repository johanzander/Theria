"""Every parameter and weight the plan runs on, in one list with what it is and where it comes from.

Sources: `fitted` (from history), `prior` (a starting value until a fit is accepted), `learned` (adjusted
by the running system), `setting` (chosen by Johan, editable on the Settings page), `constant` (fixed in
the code), `derived` (computed from the others).
"""

from typing import Any

from core.theria.plan import disturbance, skill, trim
from core.theria.plan.night_drift import NightDrift
from core.theria.plan.plan_model import Plan
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.room_model import RoomParams
from core.theria.plan.settings import PlanSettings
from core.theria.plan.warmup import WarmupFit


def _entry(
    group: str,
    key: str,
    label: str,
    value: Any,
    unit: str,
    source: str,
    description: str,
) -> dict[str, Any]:
    return {
        "group": group,
        "key": key,
        "label": label,
        "value": value,
        "unit": unit,
        "source": source,
        "description": description,
    }


def build_catalog(
    *,
    settings: PlanSettings,
    pump: PumpParams,
    pump_fitted: bool,
    room: RoomParams,
    room_fitted: bool,
    night: NightDrift | None,
    warmup: WarmupFit | None,
    morning_margin_c: float,
    disturbance_c_per_h: float,
    authority: float,
    skill_error_c_per_h: float | None,
    plan: Plan | None,
) -> list[dict[str, Any]]:
    p_src = "fitted" if pump_fitted else "prior"
    r_src = "fitted" if room_fitted else "prior"
    e: list[dict[str, Any]] = []

    g = "Pump: flow temperature"
    e += [
        _entry(
            g,
            "flow_intercept",
            "Flow temperature at 0 °C outdoors",
            pump.flow_intercept,
            "°C",
            p_src,
            "The heat curve: the flow temperature the pump asks for at 0 °C outdoors and the base setpoint.",
        ),
        _entry(
            g,
            "flow_slope",
            "Flow temperature per °C outdoors",
            pump.flow_slope,
            "°C/°C",
            p_src,
            "How much lower the flow temperature gets for each degree warmer it is outdoors.",
        ),
        _entry(
            g,
            "flow_per_rts",
            "Flow temperature per setpoint step",
            pump.flow_per_rts,
            "°C/°C",
            p_src,
            "Used outside the setpoints the pump has been measured at; between them the measured flow counts.",
        ),
        _entry(
            g,
            "rts_ref",
            "Setpoint the curve refers to",
            pump.rts_ref,
            "°C",
            p_src,
            "The setpoint at which the intercept above was measured.",
        ),
    ]
    g = "Pump: electricity"
    e += [
        _entry(
            g,
            "energy_intercept",
            "Electricity at 0 °C and setpoint 0",
            pump.energy_intercept,
            "kWh/h",
            p_src,
            "Heating electricity per hour is intercept + outdoor term + setpoint term (never below zero).",
        ),
        _entry(
            g,
            "energy_per_outdoor",
            "Electricity per °C outdoors",
            pump.energy_per_outdoor,
            "kWh/h per °C",
            p_src,
            "Colder outdoors makes the pump use more (negative number).",
        ),
        _entry(
            g,
            "energy_per_rts",
            "Electricity per setpoint step",
            pump.energy_per_rts,
            "kWh/h per °C",
            p_src,
            "What one degree of setpoint costs in electricity: the lever the plan saves money with.",
        ),
        _entry(
            g,
            "aux_onset_c",
            "Electric heater starts above flow",
            pump.aux_onset_c,
            "°C",
            p_src,
            "Above this flow temperature the electric heater helps and the electricity gets dear.",
        ),
        _entry(
            g,
            "aux_per_flow_c",
            "Electric heater per °C above the start",
            pump.aux_per_flow_c,
            "kWh/h per °C",
            p_src,
            "How much the electric heater draws for each degree of flow above its start.",
        ),
    ]
    g = "Room: response to the flow"
    gain_total = room.k_slab + room.k_out
    gain = room.k_slab / gain_total if gain_total > 0 else 0.0
    e += [
        _entry(
            g,
            "tau_slab_h",
            "Floor lag",
            room.tau_slab_h,
            "h",
            "constant",
            "How fast the floor follows the flow temperature (one hour is fixed).",
        ),
        _entry(
            g,
            "k_slab",
            "Floor to room coupling",
            room.k_slab,
            "1/h",
            r_src,
            "Share of the gap between floor and room that moves into the room each hour (valves open).",
        ),
        _entry(
            g,
            "k_out",
            "Loss to the outside",
            room.k_out,
            "1/h",
            r_src,
            "Share of the gap between room and outdoors that the room loses each hour.",
        ),
        _entry(
            g,
            "room_gain_c_per_c",
            "Room response per °C of flow",
            gain,
            "°C/°C",
            "derived",
            "How far the room moves in the long run for one degree of flow (coupling / (coupling + loss)).",
        ),
        _entry(
            g,
            "room_time_constant_days",
            "Room time constant",
            (1.0 / gain_total / 24.0) if gain_total > 0 else None,
            "days",
            "derived",
            "How long the room takes to close most of a gap: the reason a cut costs so little for hours.",
        ),
    ]
    g = "Room: the sun"
    e += [
        _entry(
            g,
            "sun_swing_c_per_wm2",
            "Sun swing",
            room.sun_swing_c_per_wm2,
            "°C per W/m²",
            r_src,
            "How much the filtered sunshine lifts the room above its level. A swing on top of the slow level.",
        ),
        _entry(
            g,
            "sun_lag_h",
            "Sun lag",
            room.sun_lag_h,
            "h",
            r_src,
            "How long after the sun the room peaks (the irradiation is smoothed over this time).",
        ),
        _entry(
            g,
            "k_sun",
            "Sun as a heating rate",
            room.k_sun,
            "°C/h per W/m²",
            "constant",
            "Not used (zero): in a slow room a heating rate from the sun would pile up for days.",
        ),
    ]
    g = "Room: night and morning"
    if night is not None:
        e += [
            _entry(
                g,
                "night_intercept_c",
                "Night change at 0 °C outdoors",
                night.intercept_c,
                "°C",
                "fitted",
                "How much the room changes over a night at 0 °C outdoors and the reference level.",
            ),
            _entry(
                g,
                "night_slope_c_per_c",
                "Night: less cooling per °C warmer",
                night.slope_c_per_c,
                "°C per °C",
                "fitted",
                "How much less the room cools over the night for each degree warmer it is outdoors.",
            ),
            _entry(
                g,
                "night_level_c_per_c",
                "Night: extra per °C warmer room",
                night.level_c_per_c,
                "°C per °C",
                "fitted",
                "A warmer room loses more over the night (negative number).",
            ),
        ]
    if warmup is not None:
        e += [
            _entry(
                g,
                "warmup_scale",
                "Warm-up speed: measured / model",
                warmup.scale,
                "",
                "fitted",
                "How the house warms up compared with the model (1 = as the model says).",
            ),
            _entry(
                g,
                "applied_scale",
                "Warm-up scale in use",
                warmup.applied_scale,
                "",
                "learned",
                "What the plan uses: 1 unless the difference is clear, and limited in steps per day.",
            ),
        ]
    e.append(
        _entry(
            g,
            "morning_margin_c",
            "Night margin",
            morning_margin_c,
            "°C",
            "learned",
            "How much warmer the night is kept, learned from mornings the room was too cool at 06:00.",
        )
    )
    g = "Running corrections"
    e += [
        _entry(
            g,
            "disturbance_c_per_h",
            "Room drift the model missed",
            disturbance_c_per_h,
            "°C/h",
            "learned",
            "What the room did beyond the model in the last hours, from measured flow, valves and weather. Fades with time. Shown, not planned with: it made the forecast worse.",
        ),
        _entry(
            g,
            "authority",
            "Authority",
            authority,
            "",
            "learned",
            "How much of the setpoint range the plan may use (0.5 to 1), earned by predicting the room well in dark hours.",
        ),
        _entry(
            g,
            "skill_error_c_per_h",
            "Systematic prediction error",
            skill_error_c_per_h,
            "°C/h",
            "learned",
            "The mean of six-hour average errors in the hours without sun; none until enough evidence.",
        ),
    ]
    g = "Comfort and limits"
    for key, label, value, unit, text in (
        (
            "floor_c",
            "Comfort floor (day)",
            settings.floor_c,
            "°C",
            "The room is not allowed below this in the daytime.",
        ),
        (
            "target_c",
            "Comfort target (day)",
            settings.target_c,
            "°C",
            "Where the dashboard and the morning check expect the room.",
        ),
        (
            "night_floor_c",
            "Comfort floor (night)",
            settings.night_floor_c,
            "°C",
            "The floor at night.",
        ),
        (
            "night_target_c",
            "Comfort target (night)",
            settings.night_target_c,
            "°C",
            "The target at night.",
        ),
        (
            "lk_base_c",
            "LK thermostat setpoint (otherwise)",
            settings.lk_base_c,
            "°C",
            "Where they stand when Theria does not control them.",
        ),
        (
            "floor_mix_max_c",
            "Floor mixing valve limit",
            settings.floor_mix_max_c,
            "°C",
            "The most flow temperature the floor loops receive.",
        ),
        (
            "rts_min_c",
            "Lowest pump setpoint",
            settings.rts_min_c,
            "°C",
            "The plan never goes below this.",
        ),
        (
            "rts_max_c",
            "Highest pump setpoint",
            settings.rts_max_c,
            "°C",
            "The plan never goes above this.",
        ),
        (
            "rts_base_c",
            "Base pump setpoint",
            settings.rts_base_c,
            "°C",
            "The setpoint the flat baseline holds.",
        ),
        (
            "min_authority",
            "Least share of the setpoint range",
            settings.min_authority,
            "",
            "The plan always has at least this share of the setpoint range; it grows to all of it as the model proves itself.",
        ),
        (
            "max_rts_step_per_hour_c",
            "Largest rise per hour",
            settings.max_rts_step_per_hour_c,
            "°C",
            "How fast the setpoint may be raised.",
        ),
    ):
        e.append(_entry(g, key, label, value, unit, "setting", text))
    g = "Estimators and limits on learning"
    e += [
        _entry(
            g,
            "disturbance_gain",
            "Disturbance gain",
            disturbance.GAIN,
            "",
            "constant",
            "Share of a new error taken into the drift estimate.",
        ),
        _entry(
            g,
            "disturbance_half_life_h",
            "Disturbance half-life",
            disturbance.HALF_LIFE_H,
            "h",
            "constant",
            "How fast the drift estimate fades.",
        ),
        _entry(
            g,
            "disturbance_max_c_per_h",
            "Largest drift estimate",
            disturbance.MAX_DRIFT_C_PER_H,
            "°C/h",
            "constant",
            "Sanity limit.",
        ),
        _entry(
            g,
            "skill_good_c_per_h",
            "Error for full authority",
            skill.GOOD_C_PER_H,
            "°C/h",
            "constant",
            "At or below this the plan has the whole setpoint range.",
        ),
        _entry(
            g,
            "skill_bad_c_per_h",
            "Error for minimum authority",
            skill.BAD_C_PER_H,
            "°C/h",
            "constant",
            "At or above this the plan has only its least share of the range.",
        ),
        _entry(
            g,
            "trim_gain_step",
            "Room gain: largest change per day",
            trim.MAX_GAIN_STEP,
            "°C/°C",
            "constant",
            "A refit may move the response this far per day.",
        ),
        _entry(
            g,
            "trim_rate_step",
            "Room rate: largest change per day",
            trim.MAX_RATE_STEP_PER_DAY,
            "per day",
            "constant",
            "The same for how fast the room closes a gap.",
        ),
        _entry(
            g,
            "trim_swing_step",
            "Sun swing: largest change per day",
            trim.MAX_SWING_STEP,
            "°C per W/m²",
            "constant",
            "The same for the sun.",
        ),
    ]
    return e
