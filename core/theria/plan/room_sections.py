"""The house as the dashboard shows it: sections (floors, salon, guest house) with their rooms and thermostats.

Display only: the plan does not read these. Each section's rooms are drawn on a chart of their own, and
its thermostats' setpoints and heat demand beside them, so that the whole house can be seen, not only
the zones the plan moves.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from core.theria.exceptions import ConfigurationError
from core.theria.plan.calibration_data import HistorySource
from core.theria.plan.series import FloatArray, Point, to_hourly

# Many room sensors (Verisure, the upper floor) and the Netatmo demand only report on change: a value
# holds for a few silent hours, after that the line breaks (a dead sensor does not draw for long).
ROOM_FFILL_HOURS = 6
DEMAND_FFILL_HOURS = 3
SETPOINT_FFILL_HOURS = (
    24  # a setpoint is logged only when it changes; read with that much lead-in
)


@dataclass(frozen=True)
class RoomPoint:
    name: str
    entity: str  # a temperature sensor


@dataclass(frozen=True)
class ThermostatPoint:
    name: str
    entity: str  # a climate entity: its `temperature` attribute is the setpoint
    demand_entity: str  # heat demand in percent
    demand_field: str | None = (
        None  # an attribute of demand_entity, or None for its state
    )


@dataclass(frozen=True)
class RoomSection:
    id: str
    name: str
    rooms: tuple[RoomPoint, ...]
    thermostats: tuple[ThermostatPoint, ...] = ()

    def __post_init__(self) -> None:
        names = [r.name for r in self.rooms]
        if len(set(names)) != len(names):
            raise ConfigurationError(f"room names repeat in section {self.id}")
        names = [t.name for t in self.thermostats]
        if len(set(names)) != len(names):
            raise ConfigurationError(f"thermostat names repeat in section {self.id}")


def room_sections_from_options(options: dict[str, Any]) -> tuple[RoomSection, ...]:
    """The `room_sections` option; empty when it is not set."""
    try:
        sections = tuple(
            RoomSection(
                id=s["id"],
                name=s["name"],
                rooms=tuple(RoomPoint(r["name"], r["entity"]) for r in s["rooms"]),
                thermostats=tuple(
                    ThermostatPoint(
                        t["name"],
                        t["entity"],
                        t["demand_entity"],
                        t.get("demand_field"),
                    )
                    for t in s.get("thermostats", [])
                ),
            )
            for s in options.get("room_sections", [])
        )
    except (KeyError, TypeError) as e:
        raise ConfigurationError(f"invalid room_sections: {e}") from e
    ids = [s.id for s in sections]
    if len(set(ids)) != len(ids):
        raise ConfigurationError("room_sections ids repeat")
    return sections


SILENT_STATES = ("unavailable", "unknown")


def silent_entities(
    states: Iterable[Mapping[str, Any]], entities: Iterable[str]
) -> frozenset[str]:
    """The entities Home Assistant has as unavailable or unknown now (a flat battery, a lost device).

    InfluxDB keeps no point for such a state, so without this a silent sensor's last value would be
    carried on as if it still held. When it fell silent is taken from its last report, not from Home
    Assistant, whose `last_changed` starts over when the integration is reloaded.
    """
    wanted = set(entities)
    return frozenset(
        s["entity_id"]
        for s in states
        if s["entity_id"] in wanted and s["state"] in SILENT_STATES
    )


def last_report(points: Sequence[Point]) -> datetime | None:
    return max((moment for moment, _ in points), default=None)


def end_silent(
    values: FloatArray, first: datetime, points: Sequence[Point], silent: bool
) -> FloatArray:
    """A silent entity's values end with the hour of its last report; nothing at all without one."""
    if not silent:
        return values
    last = last_report(points)
    if last is None:
        return np.full(len(values), np.nan)
    return cut_after(values, first, last)


def cut_after(
    values: FloatArray, first: datetime, since: datetime | None
) -> FloatArray:
    """NaN from the first hour that starts after `since`; the hour it fell silent in keeps its reports."""
    if since is None:
        return values
    out = np.array(values, dtype=float)
    hours_before = (since - first) / timedelta(hours=1)
    out[max(0, int(np.floor(hours_before)) + 1) :] = np.nan
    return out


def section_entities(sections: Sequence[RoomSection]) -> set[str]:
    """Every entity the sections read."""
    return {
        e
        for s in sections
        for e in [r.entity for r in s.rooms]
        + [t.entity for t in s.thermostats]
        + [t.demand_entity for t in s.thermostats]
    }


@dataclass(frozen=True)
class SectionMeasured:
    rooms: dict[str, FloatArray]
    setpoints: dict[str, FloatArray]
    demand: dict[str, FloatArray]
    # rooms and thermostats Home Assistant has lost now, with their last report (None: none in the window)
    offline: dict[str, datetime | None] = field(default_factory=dict)


def _hourly(
    source: HistorySource,
    entity: str,
    field: str | None,
    first: datetime,
    stop: datetime,
    n: int,
    ffill_hours: int,
    silent: frozenset[str],
    lead_hours: int = 0,
    how: str = "mean",
) -> FloatArray:
    """An entity's hourly values from `first`, read with `lead_hours` of lead-in, ended when it is silent."""
    lead = timedelta(hours=lead_hours)
    points = source.series(entity, first - lead, stop, field)
    values = to_hourly(points, first - lead, n + lead_hours, how, ffill_hours)
    return end_silent(values[lead_hours:], first, points, entity in silent)


def load_section_measurements(
    source: HistorySource,
    sections: Sequence[RoomSection],
    first: datetime,
    stop: datetime,
    n: int,
    silent: frozenset[str] = frozenset(),
) -> dict[str, SectionMeasured]:
    """Hourly room temperatures, thermostat setpoints and heat demand per section, from `first` for n hours.

    `silent` (from `silent_entities`) are the entities Home Assistant has lost: they end with their
    last report, and their rooms and thermostats are listed as offline with it.
    """

    def offline(
        entities: Iterable[tuple[str, str, str | None, int]],
    ) -> dict[str, datetime | None]:
        out: dict[str, datetime | None] = {}
        for name, entity, attribute, lead in entities:
            if entity in silent and name not in out:
                lead_td = timedelta(hours=lead)
                out[name] = last_report(
                    source.series(entity, first - lead_td, stop, attribute)
                )
        return out

    return {
        s.id: SectionMeasured(
            rooms={
                r.name: _hourly(
                    source, r.entity, None, first, stop, n, ROOM_FFILL_HOURS, silent
                )
                for r in s.rooms
            },
            setpoints={
                t.name: _hourly(
                    source,
                    t.entity,
                    "temperature",
                    first,
                    stop,
                    n,
                    SETPOINT_FFILL_HOURS,
                    silent,
                    lead_hours=SETPOINT_FFILL_HOURS,
                    how="last",
                )
                for t in s.thermostats
            },
            demand={
                t.name: _hourly(
                    source,
                    t.demand_entity,
                    t.demand_field,
                    first,
                    stop,
                    n,
                    DEMAND_FFILL_HOURS,
                    silent,
                )
                for t in s.thermostats
            },
            offline=offline(
                [(r.name, r.entity, None, 0) for r in s.rooms]
                + [
                    (t.name, t.entity, "temperature", SETPOINT_FFILL_HOURS)
                    for t in s.thermostats
                ]
            ),
        )
        for s in sections
    }


def sections_meta(sections: Sequence[RoomSection]) -> list[dict[str, Any]]:
    """What the charts need to lay out a section: its name and the names of its rooms and thermostats."""
    return [
        {
            "id": s.id,
            "name": s.name,
            "rooms": [r.name for r in s.rooms],
            "thermostats": [t.name for t in s.thermostats],
        }
        for s in sections
    ]
