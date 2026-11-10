"""The house as the dashboard shows it: sections (floors, salon, guest house) with their rooms and thermostats.

Display only: the plan does not read these. Each section's rooms are drawn on a chart of their own, and
its thermostats' setpoints and heat demand beside them, so that the whole house can be seen, not only
the zones the plan moves.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from core.theria.exceptions import ConfigurationError
from core.theria.plan.calibration_data import HistorySource
from core.theria.plan.series import FloatArray, to_hourly

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


@dataclass(frozen=True)
class SectionMeasured:
    rooms: dict[str, FloatArray]
    setpoints: dict[str, FloatArray]
    demand: dict[str, FloatArray]


def _setpoint(
    source: HistorySource, entity: str, first: datetime, stop: datetime, n: int
) -> FloatArray:
    lead = timedelta(hours=SETPOINT_FFILL_HOURS)
    values = to_hourly(
        source.series(entity, first - lead, stop, "temperature"),
        first - lead,
        n + SETPOINT_FFILL_HOURS,
        how="last",
        ffill_hours=SETPOINT_FFILL_HOURS,
    )
    return values[SETPOINT_FFILL_HOURS:]


def load_section_measurements(
    source: HistorySource,
    sections: Sequence[RoomSection],
    first: datetime,
    stop: datetime,
    n: int,
) -> dict[str, SectionMeasured]:
    """Hourly room temperatures, thermostat setpoints and heat demand per section, from `first` for n hours."""
    return {
        s.id: SectionMeasured(
            rooms={
                r.name: to_hourly(
                    source.series(r.entity, first, stop),
                    first,
                    n,
                    ffill_hours=ROOM_FFILL_HOURS,
                )
                for r in s.rooms
            },
            setpoints={
                t.name: _setpoint(source, t.entity, first, stop, n)
                for t in s.thermostats
            },
            demand={
                t.name: to_hourly(
                    source.series(t.demand_entity, first, stop, t.demand_field),
                    first,
                    n,
                    ffill_hours=DEMAND_FFILL_HOURS,
                )
                for t in s.thermostats
            },
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
