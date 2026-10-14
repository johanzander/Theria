"""The 15-minute regulator: each ground-floor room is watched on its measured temperature.

The LK thermostat, held at the room limit, already closes a room's valves when the room reaches
it. The floor keeps releasing heat afterwards, so the room rises past the limit; the regulator
learns how far (from the events when the measured valves closed) and closes a room's valves that
much earlier. It works on the measured temperature and the rate it is moving at, whatever the
cause (sun, stove, the plan), and does not read the plan.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from core.theria.plan.settings import RoomControl

HYSTERESIS_C = 0.3  # a closed room reopens this far below the level it was closed at
MIN_MARGIN_C = 0.1  # the least it closes ahead of the limit
RISING_C_PER_H = 0.02  # below this a room at the level is not climbing
MAX_RISE_C = 2.0  # the most the learned rise may be
RISE_MEMORY = 0.5  # weight of the earlier rise when a new one arrives
NEAR_LIMIT_C = 0.7  # a closing counts as an event when the room is this close to the limit
FALLING_C = 0.1  # the event ends when the room is this far below its peak
EVENT_TIMEOUT = timedelta(hours=3)
RATE_WINDOW = (timedelta(minutes=5), timedelta(hours=1))
SENSIBLE_C = (5.0, 40.0)


@dataclass(frozen=True)
class RoomStatus:
    room: str
    temperature_c: float | None
    rate_c_per_h: float | None
    state: str  # open | closed | unknown
    limit_c: float
    rise_c: float
    close_at_c: float
    valve_fraction: float | None
    events: tuple[tuple[str, str], ...] = ()  # (action, detail) of what changed this cycle


class RoomRegulator:
    def __init__(
        self, room: str, limit_c: float, rise_c: float = 0.0, events: int = 0
    ) -> None:
        self.room = room
        self.limit_c = limit_c
        self.rise_c = rise_c
        self.events = events
        self.closed = False
        self._previous: tuple[datetime, float] | None = None
        self._was_closed_by_valves = False
        self._event: tuple[datetime, float, float] | None = None  # start, closed at, peak

    def export_state(self) -> dict[str, Any]:
        return {"rise_c": self.rise_c, "events": self.events}

    def load_state(self, state: Mapping[str, Any]) -> None:
        self.rise_c = float(state["rise_c"])
        self.events = int(state["events"])

    def update(
        self, now: datetime, temperature_c: float | None, valve_fraction: float | None
    ) -> RoomStatus:
        close_at = self.limit_c - max(self.rise_c, MIN_MARGIN_C)
        if temperature_c is None:
            return RoomStatus(
                self.room, None, None, "unknown", self.limit_c, self.rise_c, close_at,
                valve_fraction,
            )
        rate = self._rate(now, temperature_c)
        events: list[tuple[str, str]] = []
        if not self.closed:
            at_level = temperature_c >= close_at
            climbing = rate is not None and rate > RISING_C_PER_H
            if temperature_c >= self.limit_c or (at_level and climbing):
                self.closed = True
                events.append(("closed", f"at {temperature_c:.1f} C (level {close_at:.1f} C)"))
        elif temperature_c <= close_at - HYSTERESIS_C:
            self.closed = False
            events.append(("opened", f"at {temperature_c:.1f} C"))
        learned = self._learn(now, temperature_c, valve_fraction)
        if learned:
            events.append(("learned", learned))
        close_at = self.limit_c - max(self.rise_c, MIN_MARGIN_C)
        return RoomStatus(
            self.room,
            temperature_c,
            rate,
            "closed" if self.closed else "open",
            self.limit_c,
            self.rise_c,
            close_at,
            valve_fraction,
            tuple(events),
        )

    def _rate(self, now: datetime, temperature_c: float) -> float | None:
        previous = self._previous
        rate = None
        if previous is not None:
            elapsed = now - previous[0]
            if RATE_WINDOW[0] <= elapsed <= RATE_WINDOW[1]:
                rate = (temperature_c - previous[1]) / (elapsed / timedelta(hours=1))
        if previous is None or now - previous[0] >= RATE_WINDOW[0]:
            self._previous = (now, temperature_c)
        return rate

    def _learn(
        self, now: datetime, temperature_c: float, valve_fraction: float | None
    ) -> str | None:
        if valve_fraction is None:
            return None
        valves_closed = valve_fraction == 0.0
        if (
            valves_closed
            and not self._was_closed_by_valves
            and self._event is None
            and temperature_c >= self.limit_c - NEAR_LIMIT_C
        ):
            self._event = (now, temperature_c, temperature_c)
        self._was_closed_by_valves = valves_closed
        if self._event is None:
            return None
        started, closed_at, peak = self._event
        peak = max(peak, temperature_c)
        self._event = (started, closed_at, peak)
        if temperature_c >= peak - FALLING_C and now - started < EVENT_TIMEOUT:
            return None
        self._event = None
        rise = min(peak - closed_at, MAX_RISE_C)
        self.rise_c = (
            rise
            if self.events == 0
            else min(RISE_MEMORY * self.rise_c + (1 - RISE_MEMORY) * rise, MAX_RISE_C)
        )
        self.events += 1
        return f"the room rose {peak - closed_at:.1f} C after the valves closed"


class Regulator:
    """All rooms: reads the sensors and the actuators, updates each room, and says what to hold."""

    def __init__(
        self,
        controls: Sequence[RoomControl],
        limit_c: float,
        actuators: Mapping[str, Sequence[str]],
    ) -> None:
        self._controls = tuple(controls)
        self._actuators = {room: tuple(entities) for room, entities in actuators.items()}
        self._rooms = {c.room: RoomRegulator(c.room, limit_c) for c in self._controls}

    def step(
        self, now: datetime, read_state: Callable[[str], str]
    ) -> list[RoomStatus]:
        statuses = []
        for control in self._controls:
            temperature = _mean_of_valid(control.temperature_entities, read_state)
            fraction = self._valve_fraction(control.room, read_state)
            statuses.append(self._rooms[control.room].update(now, temperature, fraction))
        return statuses

    @property
    def rooms(self) -> list[RoomRegulator]:
        return list(self._rooms.values())

    def holds(self, statuses: Sequence[RoomStatus]) -> dict[str, float]:
        """Thermostat setpoints that close the rooms that should be closed (half a degree steps)."""
        thermostats = {c.room: c.thermostat_entity for c in self._controls}
        return {
            thermostats[s.room]: math.floor((s.temperature_c - 0.5) * 2) / 2
            for s in statuses
            if s.state == "closed" and s.temperature_c is not None
        }

    def _valve_fraction(
        self, room: str, read_state: Callable[[str], str]
    ) -> float | None:
        entities = self._actuators.get(room, ())
        if not entities:
            return None
        try:
            states = [read_state(e) for e in entities]
        except (RuntimeError, ValueError, KeyError, TypeError):
            return None
        if any(state not in ("on", "off") for state in states):
            return None
        return sum(state == "on" for state in states) / len(states)

    def export_state(self) -> dict[str, Any]:
        return {room: r.export_state() for room, r in self._rooms.items()}

    def load_state(self, state: Mapping[str, Any]) -> None:
        for room, saved in state.items():
            if room in self._rooms:
                self._rooms[room].load_state(saved)


def _mean_of_valid(
    entities: Sequence[str], read_state: Callable[[str], str]
) -> float | None:
    values = []
    for entity in entities:
        try:
            value = float(read_state(entity))
        except (RuntimeError, ValueError, KeyError, TypeError):
            continue
        if np.isfinite(value) and SENSIBLE_C[0] <= value <= SENSIBLE_C[1]:
            values.append(value)
    return float(np.mean(values)) if values else None
