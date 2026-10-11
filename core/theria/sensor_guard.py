"""Guardrails for the comfort temperature sensors.

Watches the sensors the plan relies on (LK thermostats and Sonoff second opinions) for the
failures that are easy to miss: the value stops changing, the battery runs low, the signal is
weak, the value is impossible, or two sensors in the same room stop agreeing. Healthy sensors
are combined into one room temperature for the plan; persistent problems become Home Assistant
notifications. Reads only; the one thing it writes is the notification.
"""

import asyncio
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import numpy as np

from core.theria.exceptions import ConfigurationError, SensorError
from core.theria.options import load_options
from core.theria.plan.series import FloatArray, Point, hour_floor, to_hourly

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_WARNING = "warning"
STATUS_FAULT = "fault"
_SEVERITY_RANK = {STATUS_OK: 0, STATUS_WARNING: 1, STATUS_FAULT: 2}

HISTORY_DAYS = 7
RECENT_HOURS = 6
MIN_OFFSET_PAIRS = (
    24  # hourly pairs needed to know a sensor's usual difference to the reference
)
MIN_RECENT_PAIRS = 3
MAX_REPORT_AGE = timedelta(minutes=5)
# Problems worth a notification; a flapping weak signal is shown in the UI only.
NOTIFIED_CODES = ("unavailable", "silent", "battery_low", "implausible", "disagree")


@dataclass(frozen=True)
class SensorSpec:
    name: str
    room: str
    temperature_entity: str
    battery_entity: str | None = None
    signal_entity: str | None = None
    reference: bool = False  # the sensor LK regulates on; others are corrected to it
    silent_warn_minutes: int = 480  # no new value for this long: warning
    silent_fault_minutes: int = 960  # ... fault


@dataclass(frozen=True)
class GuardSettings:
    sensors: tuple[SensorSpec, ...]
    notify: bool = True
    battery_warn_percent: float = 25.0
    battery_fault_percent: float = 10.0
    signal_warn_dbm: float = -85.0
    agreement_tolerance_c: float = 1.0
    plausible_min_c: float = -10.0
    plausible_max_c: float = 45.0
    confirm_checks: int = (
        2  # a problem must be seen in this many checks in a row to notify
    )

    def __post_init__(self) -> None:
        if not self.sensors:
            raise ConfigurationError("sensor_guard needs at least one sensor")
        for spec in self.sensors:
            if spec.silent_warn_minutes >= spec.silent_fault_minutes:
                raise ConfigurationError(
                    f"{spec.name}: silent_warn must be below silent_fault"
                )
        if self.battery_fault_percent >= self.battery_warn_percent:
            raise ConfigurationError(
                "battery_fault_percent must be below battery_warn_percent"
            )
        if self.confirm_checks < 1:
            raise ConfigurationError("confirm_checks must be at least 1")
        rooms = {spec.room for spec in self.sensors}
        for room in rooms:
            references = [s for s in self.sensors if s.room == room and s.reference]
            if len(references) != 1:
                raise ConfigurationError(
                    f"room {room!r} needs exactly one reference sensor, found {len(references)}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GuardSettings":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ConfigurationError(
                f"unknown sensor_guard settings: {sorted(unknown)}"
            )
        raw_sensors = data.get("sensors")
        if not raw_sensors:
            raise ConfigurationError("sensor_guard needs at least one sensor")
        spec_fields = {f.name for f in fields(SensorSpec)}
        specs = []
        for raw in raw_sensors:
            extra = set(raw) - spec_fields
            if extra:
                raise ConfigurationError(f"unknown sensor settings: {sorted(extra)}")
            try:
                specs.append(SensorSpec(**raw))
            except TypeError as e:
                raise ConfigurationError(f"invalid sensor definition {raw}: {e}") from e
        rest = {k: v for k, v in data.items() if k != "sensors"}
        try:
            return cls(sensors=tuple(specs), **rest)
        except TypeError as e:
            raise ConfigurationError(f"invalid sensor_guard settings: {e}") from e

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "GuardSettings | None":
        """Settings from the add-on options, or None when there is no `sensor_guard` section."""
        section = options.get("sensor_guard")
        return None if section is None else cls.from_dict(section)


def load_guard_settings(options_path: str, config_path: str) -> GuardSettings | None:
    """Load the `sensor_guard` section (options.json in production, config.yaml in development)."""
    options = load_options(options_path, config_path)
    return None if options is None else GuardSettings.from_options(options)


@dataclass(frozen=True)
class EntityState:
    """What the guard needs from a Home Assistant state."""

    state: str
    last_changed: datetime | None


@dataclass(frozen=True)
class Issue:
    code: str
    severity: str  # STATUS_WARNING or STATUS_FAULT
    message: str


@dataclass(frozen=True)
class SensorReport:
    name: str
    room: str
    temperature_entity: str
    status: str
    issues: tuple[Issue, ...]
    value_c: float | None
    age_minutes: float | None  # since the value last changed
    battery_percent: float | None
    signal_dbm: float | None
    offset_c: float | None  # usual difference to the room's reference sensor
    reference: bool
    checked_at: datetime
    usable: bool  # has a valid current reading; a low battery alone does not disqualify


def _number(entity: EntityState | None) -> float | None:
    if entity is None:
        return None
    try:
        value = float(entity.state)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def evaluate_sensor(
    spec: SensorSpec,
    settings: GuardSettings,
    temperature: EntityState,
    battery: EntityState | None,
    signal: EntityState | None,
    now: datetime,
) -> SensorReport:
    """Judge one sensor from its current states (no history needed)."""
    issues: list[Issue] = []
    value = _number(temperature)
    age = (
        None
        if temperature.last_changed is None
        else (now - temperature.last_changed).total_seconds() / 60.0
    )
    battery_percent = _number(battery)
    signal_dbm = _number(signal)

    if value is None:
        # HA only knows since when it has *seen* the sensor unavailable (a restart resets it)
        since = "" if age is None else f"; HA har sett det i {_format_age(age)}"
        issues.append(
            Issue(
                "unavailable",
                STATUS_FAULT,
                f"rapporterar inte ({temperature.state}){since}",
            )
        )
    else:
        if not settings.plausible_min_c <= value <= settings.plausible_max_c:
            issues.append(
                Issue("implausible", STATUS_FAULT, f"orimligt värde {value:.1f} °C")
            )
        if age is not None and age >= spec.silent_fault_minutes:
            issues.append(
                Issue("silent", STATUS_FAULT, f"inget nytt värde på {_format_age(age)}")
            )
        elif age is not None and age >= spec.silent_warn_minutes:
            issues.append(
                Issue(
                    "silent", STATUS_WARNING, f"inget nytt värde på {_format_age(age)}"
                )
            )
    if battery_percent is not None:
        if battery_percent <= settings.battery_fault_percent:
            issues.append(
                Issue(
                    "battery_low",
                    STATUS_FAULT,
                    f"batteriet är på {battery_percent:.0f} %",
                )
            )
        elif battery_percent <= settings.battery_warn_percent:
            issues.append(
                Issue(
                    "battery_low",
                    STATUS_WARNING,
                    f"batteriet är på {battery_percent:.0f} %",
                )
            )
    if signal_dbm is not None and signal_dbm <= settings.signal_warn_dbm:
        issues.append(
            Issue("signal_weak", STATUS_WARNING, f"svag signal ({signal_dbm:.0f} dBm)")
        )

    return SensorReport(
        name=spec.name,
        room=spec.room,
        temperature_entity=spec.temperature_entity,
        status=_worst(issues),
        issues=tuple(issues),
        value_c=value,
        age_minutes=age,
        battery_percent=battery_percent,
        signal_dbm=signal_dbm,
        offset_c=None,
        reference=spec.reference,
        checked_at=now,
        usable=_usable(value, issues),
    )


def _usable(value: float | None, issues: list[Issue]) -> bool:
    """A reading can be used unless the sensor is unavailable, impossible or silent (fault)."""
    if value is None:
        return False
    return not any(
        i.code in ("unavailable", "implausible")
        or (i.code == "silent" and i.severity == STATUS_FAULT)
        for i in issues
    )


def _worst(issues: list[Issue]) -> str:
    worst = STATUS_OK
    for issue in issues:
        if _SEVERITY_RANK[issue.severity] > _SEVERITY_RANK[worst]:
            worst = issue.severity
    return worst


def _format_age(minutes: float) -> str:
    if minutes < 90:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    return f"{hours:.0f} h" if hours < 48 else f"{hours / 24:.0f} dygn"


def _reference_of(settings: GuardSettings, room: str) -> SensorSpec | None:
    return next((s for s in settings.sensors if s.room == room and s.reference), None)


def room_offsets(
    settings: GuardSettings, hourly_by_entity: dict[str, FloatArray]
) -> dict[str, float]:
    """Usual difference to the room's reference sensor (median over the history).

    The reference itself is 0.0. A sensor without enough overlapping hours gets no entry.
    """
    offsets: dict[str, float] = {}
    for spec in settings.sensors:
        reference = _reference_of(settings, spec.room)
        if reference is None:
            continue
        if spec.reference:
            offsets[spec.temperature_entity] = 0.0
            continue
        ref = hourly_by_entity.get(reference.temperature_entity)
        own = hourly_by_entity.get(spec.temperature_entity)
        if ref is None or own is None:
            continue
        both = np.isfinite(ref) & np.isfinite(own)
        if int(both.sum()) >= MIN_OFFSET_PAIRS:
            offsets[spec.temperature_entity] = float(np.median(own[both] - ref[both]))
    return offsets


class Notifier(Protocol):
    def notify(self, notification_id: str, title: str, message: str) -> None: ...

    def dismiss(self, notification_id: str) -> None: ...


class StateSource(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def get_numeric_history(
        self, entity_id: str, start: datetime, stop: datetime
    ) -> list[Point]: ...


def _entity_state(source: StateSource, entity_id: str | None) -> EntityState | None:
    if entity_id is None:
        return None
    try:
        raw = source.get_state(entity_id)
    except (ValueError, RuntimeError):
        return EntityState("unavailable", None)
    changed = raw.get("last_changed")
    return EntityState(
        str(raw.get("state")),
        (
            None
            if changed is None
            else datetime.fromisoformat(changed.replace("Z", "+00:00"))
        ),
    )


class SensorGuard:
    """Checks the comfort sensors and hands out a trustworthy room temperature."""

    def __init__(
        self,
        source: StateSource,
        settings: GuardSettings,
        notifier: Notifier | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ):
        self._source = source
        self._settings = settings
        self._notifier = notifier
        self._now = now
        self._reports: list[SensorReport] = []
        self._offsets: dict[str, float] = {}
        self._checked_at: datetime | None = None
        self._task: asyncio.Task[None] | None = None
        self._lock = threading.Lock()
        self._streak: dict[tuple[str, str], int] = {}
        self._notified: dict[tuple[str, str], str] = (
            {}
        )  # (entity, code) -> severity notified
        self._cleared_old_notifications = False

    async def cycle(self) -> None:
        """One scheduled check. Never raises: a broken check must not stop the loop."""
        try:
            await asyncio.to_thread(self.scheduled_check)
        except Exception:
            logger.exception("comfort sensor check failed")

    async def start(self, interval_seconds: int = 600) -> None:
        async def loop() -> None:
            while True:
                await self.cycle()
                await asyncio.sleep(interval_seconds)

        self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def reports(self) -> list[SensorReport]:
        return list(self._reports)

    def checked_at(self) -> datetime | None:
        return self._checked_at

    def scheduled_check(self) -> list[SensorReport]:
        """The periodic check. Only these count towards notifications, so that page loads,
        plan runs and manual rechecks cannot shorten the confirmation time."""
        with self._lock:
            reports = self._check()
            self._notify_changes(reports)
        return reports

    def check(self) -> list[SensorReport]:
        """Check now and update the status (no notifications)."""
        with self._lock:
            return self._check()

    def _check(self) -> list[SensorReport]:
        now = self._now()
        s = self._settings
        history = self._hourly_history(now)
        self._offsets = room_offsets(s, history)
        reports = []
        for spec in s.sensors:
            report = evaluate_sensor(
                spec,
                s,
                _entity_state(self._source, spec.temperature_entity)
                or EntityState("unavailable", None),
                _entity_state(self._source, spec.battery_entity),
                _entity_state(self._source, spec.signal_entity),
                now,
            )
            reports.append(report)
        reports = self._with_agreement(reports, history)
        self._reports = reports
        self._checked_at = now
        return list(reports)

    def room_temperature(self) -> float:
        """Room temperature for the plan; refreshes the status first when it is stale."""
        if self._checked_at is None or self._now() - self._checked_at > MAX_REPORT_AGE:
            self.check()
        return self.current_room_temperature()

    def current_room_temperature(self) -> float:
        """From the latest status without any new request.

        Each usable sensor is corrected to its room's reference sensor; the rooms are then
        averaged with equal weight, whatever the number of sensors in them.
        """
        by_room: dict[str, list[float]] = {}
        for report in self._reports:
            offset = self._offsets.get(report.temperature_entity)
            if not report.usable or report.value_c is None or offset is None:
                continue
            by_room.setdefault(report.room, []).append(report.value_c - offset)
        if not by_room:
            raise SensorError("no healthy comfort temperature sensor")
        return float(np.mean([np.mean(values) for values in by_room.values()]))

    def _hourly_history(self, now: datetime) -> dict[str, FloatArray]:
        rooms: dict[str, int] = {}
        for spec in self._settings.sensors:
            rooms[spec.room] = rooms.get(spec.room, 0) + 1
        end = hour_floor(now) + timedelta(hours=1)
        n = HISTORY_DAYS * 24
        start = end - timedelta(hours=n)
        out: dict[str, FloatArray] = {}
        for spec in self._settings.sensors:
            if rooms[spec.room] < 2:
                continue
            try:
                points = self._source.get_numeric_history(
                    spec.temperature_entity, start, end
                )
            except (ValueError, RuntimeError):
                continue
            out[spec.temperature_entity] = to_hourly(
                points, start, n, how="mean", ffill_hours=2
            )
        return out

    def _with_agreement(
        self, reports: list[SensorReport], history: dict[str, FloatArray]
    ) -> list[SensorReport]:
        s = self._settings
        result = []
        for report, spec in zip(reports, s.sensors, strict=True):
            offset = self._offsets.get(spec.temperature_entity)
            issues = list(report.issues)
            reference = _reference_of(s, spec.room)
            ref_series = (
                None if reference is None else history.get(reference.temperature_entity)
            )
            own_series = history.get(spec.temperature_entity)
            if (
                not spec.reference
                and reference is not None
                and offset is not None
                and ref_series is not None
                and own_series is not None
            ):
                recent_ref = ref_series[-RECENT_HOURS:]
                recent_own = own_series[-RECENT_HOURS:]
                both = np.isfinite(recent_ref) & np.isfinite(recent_own)
                if int(both.sum()) >= MIN_RECENT_PAIRS:
                    recent = float(np.median(recent_own[both] - recent_ref[both]))
                    drift = recent - offset
                    if abs(drift) > s.agreement_tolerance_c:
                        issues.append(
                            Issue(
                                "disagree",
                                STATUS_WARNING,
                                f"avviker {drift:+.1f} °C från sin vanliga skillnad mot {reference.name}",
                            )
                        )
            result.append(
                SensorReport(
                    **{
                        **report.__dict__,
                        "issues": tuple(issues),
                        "status": _worst(issues),
                        "offset_c": offset,
                    }
                )
            )
        return result

    def _notify_changes(self, reports: list[SensorReport]) -> None:
        if self._notifier is None or not self._settings.notify:
            return
        present: dict[tuple[str, str], tuple[SensorReport, Issue]] = {}
        for report in reports:
            for issue in report.issues:
                # a quiet spell in a stable room is shown, but only a fault is worth a ping
                if issue.code in NOTIFIED_CODES and not (
                    issue.code == "silent" and issue.severity == STATUS_WARNING
                ):
                    present[(report.temperature_entity, issue.code)] = (report, issue)
        if not self._cleared_old_notifications:
            # notifications from before a restart may belong to problems that are gone
            for spec in self._settings.sensors:
                for code in NOTIFIED_CODES:
                    if (spec.temperature_entity, code) not in present:
                        self._notifier.dismiss(
                            _notification_id((spec.temperature_entity, code))
                        )
            self._cleared_old_notifications = True
        for key, (report, issue) in present.items():
            self._streak[key] = self._streak.get(key, 0) + 1
            confirmed = self._streak[key] >= self._settings.confirm_checks
            if confirmed and self._notified.get(key) != issue.severity:
                self._notifier.notify(
                    _notification_id(key),
                    "Theria: problem med givare",
                    f"{report.name}: {issue.message}.",
                )
                self._notified[key] = issue.severity
                logger.warning("notified: %s %s", report.name, issue.message)
        for key in list(self._streak):
            if key not in present:
                del self._streak[key]
        for key in list(self._notified):
            if key not in present:
                self._notifier.dismiss(_notification_id(key))
                del self._notified[key]
                logger.info("notification dismissed: %s %s", key[0], key[1])


def _notification_id(key: tuple[str, str]) -> str:
    entity, code = key
    return f"theria_sensor_{entity.replace('.', '_')}_{code}"
