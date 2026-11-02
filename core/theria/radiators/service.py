"""The radiator service: every 15 minutes, plan the salon and the guest house and log what it would write.

Shadow mode only. Nothing here calls `set_temperature` or any other service: the thermostats are read,
the plan is computed and the setpoints it would write are logged next to the ones they have.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from core.theria.exceptions import PriceDataError, SensorError
from core.theria.plan.event_log import EventLog
from core.theria.plan.prices import PricePeriod, hourly_prices
from core.theria.radiators.guest_plan import GuestHour, plan_guest_room
from core.theria.radiators.salon_plan import SalonHour, SalonModel, plan_salon
from core.theria.radiators.salon_reading import SalonReading, read_salon
from core.theria.radiators.savings import (
    guest_saving_sek,
    salon_saving_sek,
    would_lift_block,
)
from core.theria.radiators.settings import RadiatorPrefs, RadiatorSettings
from core.theria.radiators.store import PrefsStore
from core.theria.sensor_guard import SensorReport

logger = logging.getLogger(__name__)

HORIZON_HOURS = 36
UNREADABLE = ("unavailable", "unknown")
OFF = "off"


def _number(value: Any) -> float | None:
    """A finite number from a Home Assistant value, or None for anything else."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


class StateSource(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...


class SalonGuard(Protocol):
    def check(self) -> list[SensorReport]: ...


@dataclass(frozen=True)
class SalonPlanView:
    reading: SalonReading
    hours: list[SalonHour]
    saving_sek: float | None
    unknown: bool


@dataclass(frozen=True)
class RadiatorSnapshot:
    computed_at: datetime
    salon: SalonPlanView
    guest: dict[str, list[GuestHour]]
    would_write: dict[str, float]  # entity -> this hour's setpoint
    read: dict[str, float | None]  # entity -> its setpoint now
    difference: dict[str, float | None]  # would write minus current
    would_lift_block: bool
    availability: dict[str, bool]
    guest_saving: tuple[float, str]
    outdoor_c: float
    outdoor_stale: bool  # the sensor could not be read: the last good value was used
    left_alone: list[str]  # thermostats set to off by hand


def _floor_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


class RadiatorService:
    """Shadow-mode planner for the radiator zones. Never writes to Home Assistant."""

    def __init__(
        self,
        ha: StateSource,
        settings: RadiatorSettings,
        store: PrefsStore,
        guard: SalonGuard,
        price_periods: Callable[[], list[PricePeriod]],
        event_log: EventLog | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        interval_seconds: int = 900,
    ):
        self._ha = ha
        self._s = settings
        self._store = store
        self._guard = guard
        self._price_periods = price_periods
        self._event_log = event_log
        self._now = now
        self._interval = interval_seconds
        self._snapshot: RadiatorSnapshot | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = threading.Lock()
        self._cycles = 0
        self._readable: dict[str, int] = {}
        self._stats_lock = threading.Lock()
        self._outdoor_c: float | None = None
        self.last_error: str | None = None

    def latest(self) -> RadiatorSnapshot | None:
        return self._snapshot

    def now(self) -> datetime:
        return self._now()

    def availability_share(self) -> dict[str, float]:
        """The share of cycles in which each thermostat could be read."""
        with self._stats_lock:
            if self._cycles == 0:
                return {}
            return {e: n / self._cycles for e, n in self._readable.items()}

    def prefs_catalog(self) -> dict[str, Any]:
        """The editable preferences with their defaults."""
        return {
            "prefs": self._store.current.to_dict(),
            "defaults": self._s.defaults.to_dict(),
            "is_default": self._store.is_default(),
        }

    def update_prefs(self, values: dict[str, Any]) -> RadiatorPrefs:
        """Check and save a change; raises ConfigurationError and changes nothing when invalid."""
        with self._lock:
            return self._store.update(values)

    def reset_prefs(self) -> RadiatorPrefs:
        with self._lock:
            return self._store.reset()

    def log(self, limit: int) -> list[dict[str, Any]]:
        """The latest shadow log lines, oldest first."""
        return [] if self._event_log is None else self._event_log.read(limit)

    def _climate(self, entity: str) -> dict[str, Any] | None:
        """A readable thermostat as numbers, or None. Malformed values become None, not errors."""
        try:
            state = self._ha.get_state(entity)
            kind = state["state"]
            attributes = state["attributes"]
        except (RuntimeError, ValueError, KeyError, TypeError):
            return None
        if kind in UNREADABLE or not isinstance(attributes, dict):
            return None
        return {
            "state": kind,
            "temperature": _number(attributes.get("temperature")),
            "current_temperature": _number(attributes.get("current_temperature")),
            "hvac_action": attributes.get("hvac_action"),
        }

    def _outdoor(self) -> tuple[float, bool]:
        """The outdoor temperature and whether it is the last good value rather than a reading."""
        try:
            reading = _number(self._ha.get_state(self._s.outdoor_entity)["state"])
        except (RuntimeError, ValueError, KeyError, TypeError):
            reading = None
        if reading is not None:
            self._outdoor_c = reading
            return reading, False
        if self._outdoor_c is None:
            raise SensorError("outdoor temperature unavailable")
        return self._outdoor_c, True

    def run_once(self) -> RadiatorSnapshot | None:
        """Compute and store one shadow plan; None when the mode is off."""
        with self._lock:
            return self._run_locked()

    def _run_locked(self) -> RadiatorSnapshot | None:
        prefs = self._store.current
        if prefs.mode == "off":
            return None
        now = self._now()
        s = self._s
        hourly = hourly_prices(self._price_periods(), _floor_hour(now))[:HORIZON_HOURS]
        if not hourly:
            raise PriceDataError("no prices for the coming hours")
        hours = [h for h, _ in hourly]
        prices = [p for _, p in hourly]
        outdoor, outdoor_stale = self._outdoor()
        t_out = [outdoor] * len(hours)

        availability: dict[str, bool] = {}
        read: dict[str, float | None] = {}
        attrs: dict[str, dict[str, Any]] = {}
        guest_entities = [r.entity for r in prefs.guest_rooms]
        for entity in (*s.salon_thermostats, *guest_entities):
            found = self._climate(entity)
            availability[entity] = found is not None
            if found is not None:
                attrs[entity] = found
            read[entity] = None if found is None else found["temperature"]
            with self._stats_lock:
                self._readable[entity] = self._readable.get(entity, 0) + (
                    1 if found is not None else 0
                )
        with self._stats_lock:
            self._cycles += 1
        left_alone = [e for e, a in attrs.items() if a["state"] == OFF]
        attrs = {e: a for e, a in attrs.items() if e not in left_alone}

        salon_view, would_write = self._salon(prefs, hours, prices, t_out, read, attrs)
        guest_plans: dict[str, list[GuestHour]] = {}
        bounds_ok = not salon_view.unknown and self._salon_above_bound(
            salon_view, prefs, now
        )
        for room in prefs.guest_rooms:
            if room.entity not in attrs:
                continue
            temperature_c = attrs[room.entity]["current_temperature"]
            band = room.band_c if room.band_c is not None else prefs.guest_band_c
            plan = plan_guest_room(hours, prices, room, band, temperature_c)
            guest_plans[room.entity] = plan
            would_write[room.entity] = plan[0].setpoint_c
            if temperature_c is not None and temperature_c < room.t_room_c - band:
                bounds_ok = False
        calling = {e: a.get("hvac_action") == "heating" for e, a in attrs.items()}
        controlled = {*s.salon_thermostats, *guest_entities}
        snapshot = RadiatorSnapshot(
            computed_at=now,
            salon=salon_view,
            guest=guest_plans,
            would_write=would_write,
            read=read,
            difference={
                e: (None if (now_c := read[e]) is None else v - now_c)
                for e, v in would_write.items()
            },
            would_lift_block=would_lift_block(calling, controlled, bounds_ok),
            availability=availability,
            guest_saving=guest_saving_sek(
                guest_plans,
                {r.entity: r.t_room_c for r in prefs.guest_rooms},
                prices,
                s.guest_loss_w_per_k,
            ),
            outdoor_c=outdoor,
            outdoor_stale=outdoor_stale,
            left_alone=left_alone,
        )
        self._snapshot = snapshot
        self._log(snapshot)
        return snapshot

    def _salon(
        self,
        prefs: RadiatorPrefs,
        hours: list[datetime],
        prices: list[float],
        t_out: list[float],
        read: dict[str, float | None],
        attrs: dict[str, dict[str, Any]],
    ) -> tuple[SalonPlanView, dict[str, float]]:
        s = self._s
        sensors = {sp.temperature_entity for sp in s.salon_sensor_guard.sensors}
        thermostat_sensors = sensors - set(s.salon_sensors)
        reading = read_salon(self._guard.check(), thermostat_sensors)
        readable = [e for e in s.salon_thermostats if e in attrs]
        if reading.mean_c is None or not readable:
            held = {e: read[e] for e in readable if read[e] is not None}
            return SalonPlanView(reading, [], None, True), {
                e: v for e, v in held.items() if v is not None
            }
        bias = (
            0.0
            if reading.thermostat_mean_c is None or reading.mean_c is None
            else reading.thermostat_mean_c - reading.mean_c
        )
        start_mean = reading.mean_c
        model = SalonModel(s.salon_c_j_per_k, s.salon_u_w_per_k, s.salon_heat_power_w)
        flags = [prefs.open_in_hour(h, s.timezone) for h in hours]
        plan = plan_salon(hours, prices, t_out, flags, start_mean, prefs, model, bias)
        saving = salon_saving_sek(plan, s.salon_baseline_c, t_out, model, start_mean)
        return SalonPlanView(reading, plan, saving, False), dict.fromkeys(
            readable, plan[0].setpoint_c
        )

    def _salon_above_bound(
        self, view: SalonPlanView, prefs: RadiatorPrefs, now: datetime
    ) -> bool:
        mean = view.reading.mean_c
        if mean is None:
            return True
        bound = (
            prefs.salon_open_target_c
            if prefs.is_open(now, self._s.timezone)
            else prefs.salon_closed_floor_c
        )
        return mean >= bound

    def _log(self, snap: RadiatorSnapshot) -> None:
        parts = [
            f"{e}: now {snap.read[e]} would {v} ({snap.difference[e]:+.1f})"
            if snap.difference[e] is not None
            else f"{e}: now unreadable would {v}"
            for e, v in snap.would_write.items()
        ]
        mean = snap.salon.reading.mean_c
        salon = (
            "salon unknown"
            if snap.salon.unknown
            else f"salon mean {mean:.2f} from {len(snap.salon.reading.used)} points"
        )
        unreadable = [e for e, ok in snap.availability.items() if not ok]
        detail = "; ".join(parts) + f" | {salon}"
        if unreadable:
            detail += f" | unreadable: {', '.join(unreadable)}"
        if snap.left_alone:
            detail += f" | set to off, left alone: {', '.join(snap.left_alone)}"
        if snap.outdoor_stale:
            detail += " | outdoor sensor unreadable, last value used"
        event = {
            "at": snap.computed_at,
            "kind": "radiators",
            "action": "shadow",
            "detail": detail,
        }
        if self._event_log is not None:
            self._event_log.append(event)

    async def cycle(self) -> None:
        """One pass. Never raises: a failing price feed or thermostat must not stop the loop."""
        try:
            await asyncio.to_thread(self.run_once)
            self.last_error = None
        except Exception as e:  # supervisor boundary: the loop must survive
            self.last_error = f"{type(e).__name__}: {e}"
            logger.exception("radiator plan failed")

    async def _loop(self) -> None:
        while True:
            await self.cycle()
            await asyncio.sleep(self._interval)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
