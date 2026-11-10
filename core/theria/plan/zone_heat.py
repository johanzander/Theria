"""What each zone needs, as far as it can be calculated, and what the dashboard shows beside it.

The heat need comes from the heat demand analysis, not from a meter: only the whole house has one, and
sun, people and the stove are not measured. See docs/ALGORITHM_EXPLAINED.md.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from core.theria.plan.calibration_data import (
    FLOW_PLAUSIBLE_C,
    MAX_HEAT_KWH_PER_HOUR,
    SMALL_GAP_HOURS,
    HistorySource,
)
from core.theria.plan.room_model import valve_open
from core.theria.plan.room_sections import (
    RoomSection,
    SectionMeasured,
    end_silent,
    load_section_measurements,
)
from core.theria.plan.series import (
    FloatArray,
    Point,
    hour_floor,
    hourly_counter_energy,
    hourly_grid,
    to_hourly,
)
from core.theria.plan.settings import PlanSettings
from core.theria.plan.shift_rules import CLOSING, OPENING, thermostat_target

GROUND_W_PER_K = 161.5  # the rooms behind the two LK thermostats (Vardagsrum 85.5, Bibliotek's rooms 76)
SALON_W_PER_K = 84.2
SALON_OTHER_HEAT_W = 250.0  # measured with the valves open: people, lighting, sun
# What the floor gives per K of flow above the room, from the LK calculation
FLOOR_W_PER_K = {"vardagsrum": 78.0, "bibliotek": 89.0}
GROUND_FLOOR_W_PER_K = sum(FLOOR_W_PER_K.values())
EXPECTED_HEATING_SHARE = 0.82  # the pump makes hot water about 18 % of the time
AFTER_HOT_WATER = timedelta(minutes=15)  # the flow is still settling
TARGET_FFILL_HOURS = 24  # the pump's target is logged only when it changes


@dataclass(frozen=True)
class ZoneMeasured:
    hours: list[datetime]
    flow_c: FloatArray
    salon_c: FloatArray
    flow_target_c: FloatArray
    heat_kwh: FloatArray  # heat the pump supplied to the heating, from its counter
    sections: dict[str, SectionMeasured] = field(default_factory=dict)


def ground_need_kw(room_c: float, t_out_c: float) -> float | None:
    """The heat the rooms behind the LK thermostats lose, and so need from somewhere: pump, floor or stove."""
    if not (math.isfinite(room_c) and math.isfinite(t_out_c)):
        return None
    return max(0.0, GROUND_W_PER_K * (room_c - t_out_c) / 1000.0)


def delivered_ground_kw(
    flow_c: float,
    room_c: float,
    open_share: float,
    heating_share: float,
    shunt_max_c: float,
) -> float | None:
    """The heat the floor loops give: only through open loops, and only while the pump heats."""
    if not all(math.isfinite(v) for v in (flow_c, room_c, open_share, heating_share)):
        return None
    above_room = max(0.0, min(flow_c, shunt_max_c) - room_c)
    return GROUND_FLOOR_W_PER_K * above_room * open_share * heating_share / 1000.0


def measured_open_share(
    open_by_room: Mapping[str, float], loops_by_room: Mapping[str, int]
) -> float:
    """The open loops as a share of the floor's output; NaN when no room is known."""
    known = [r for r in open_by_room if r in FLOOR_W_PER_K and loops_by_room.get(r)]
    if not known or not all(math.isfinite(open_by_room[r]) for r in known):
        return float("nan")
    return (
        sum(FLOOR_W_PER_K[r] * open_by_room[r] / loops_by_room[r] for r in known)
        / GROUND_FLOOR_W_PER_K
    )


def salon_need_kw(room_c: float, t_out_c: float) -> float | None:
    """The heat the salon needs to stay where it is."""
    if not (math.isfinite(room_c) and math.isfinite(t_out_c)):
        return None
    return max(0.0, (SALON_W_PER_K * (room_c - t_out_c) - SALON_OTHER_HEAT_W) / 1000.0)


def expected_open_share(
    mode: str, room_c: float, night: bool, settings: PlanSettings
) -> float:
    """The share of the floor loops the thermostats are expected to keep open."""
    if mode == OPENING:
        return float(valve_open(room_c, settings.lk_upper_c))
    target = (
        thermostat_target(CLOSING, night, None, settings)
        if mode == CLOSING
        else settings.lk_base_c
    )
    return float(valve_open(room_c, target))


def _heating_only(flow: list[Point], valve: list[Point]) -> list[Point]:
    """The flow readings outside hot-water making and the quarter of an hour after it."""
    events = sorted(valve, key=lambda p: p[0])
    spans: list[tuple[datetime, datetime]] = []
    started: datetime | None = None
    for moment, value in events:
        if value >= 0.5 and started is None:
            started = moment
        elif value < 0.5 and started is not None:
            spans.append((started, moment + AFTER_HOT_WATER))
            started = None
    return [
        (moment, value)
        for moment, value in flow
        if not any(a <= moment < b for a, b in spans)
        and (started is None or moment < started)
    ]


def _plausible_flow(points: list[Point]) -> list[Point]:
    return [p for p in points if FLOW_PLAUSIBLE_C[0] <= p[1] <= FLOW_PLAUSIBLE_C[1]]


def load_zone_measurements(
    source: HistorySource,
    settings: PlanSettings,
    salon_points: Sequence[tuple[str, str | None]],
    sections: Sequence[RoomSection],
    start: datetime,
    stop: datetime,
    silent: frozenset[str] = frozenset(),
) -> ZoneMeasured:
    """The flow to the heating, the pump's own target for it, the mean of the salon's points and
    every section's rooms and thermostats. `silent` ends the points Home Assistant has lost."""
    first = hour_floor(start)
    n = int((hour_floor(stop) - first) / timedelta(hours=1))
    flow = to_hourly(
        _heating_only(
            _plausible_flow(source.series(settings.flow_measured_entity, first, stop)),
            source.series(settings.heating_mode_entity, first, stop),
        ),
        first,
        n,
    )
    target = to_hourly(
        _plausible_flow(source.series(settings.flow_setpoint_entity, first, stop)),
        first,
        n,
        how="last",
        ffill_hours=TARGET_FFILL_HOURS,
    )

    def salon_point(entity: str, attribute: str | None) -> FloatArray:
        series = source.series(entity, first, stop, attribute)
        return end_silent(to_hourly(series, first, n), first, series, entity in silent)

    points = [salon_point(entity, attribute) for entity, attribute in salon_points]
    if points:
        stacked = np.array(points)
        seen = np.isfinite(stacked).sum(axis=0)
        total = np.nansum(stacked, axis=0)
        salon = np.divide(
            total, seen, out=np.full(n, np.nan), where=seen > 0, casting="unsafe"
        )
    else:
        salon = np.full(n, np.nan)
    return ZoneMeasured(
        hourly_grid(first, n),
        flow,
        salon,
        target,
        hourly_counter_energy(
            source.series(settings.heat_counter_entity, first, stop),
            first,
            n,
            MAX_HEAT_KWH_PER_HOUR,
            SMALL_GAP_HOURS,
        ),
        load_section_measurements(source, sections, first, stop, n, silent),
    )
