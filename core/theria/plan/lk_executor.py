"""Writing the LK room thermostat setpoints: a static limit per room.

The LK valves close when the room reaches the thermostat setpoint, so the setpoint is the room
limit (`lk_limit_c`): the floor can store heat up to it and no further. It is held there at all
times. The regulator may hold a single thermostat lower to close that room's valves for a while
(`holds`); the thermostat goes back to the limit when the hold is released.

The thermostats sit behind the LK cloud, which Home Assistant polls every 15-30 minutes
(`last_successful_cloud_fetch`): a write shows up minutes later, the value is rounded to a tenth
of a degree, and the value from before the write can come back for a while. A write that has not
shown yet is therefore pending, not a failure; the old value is not a hand change; and a value
within a tenth of the target is the target. It is a failure when it has not shown after
`CLOUD_GRACE`, and a value that is neither the old nor the written one is a hand change.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from core.theria.plan.confirm import read_until
from core.theria.plan.settings import PlanSettings

logger = logging.getLogger(__name__)

MANUAL_PAUSE = timedelta(hours=1)
MANUAL_CHANGE_C = 0.25
CONFIRM_TOLERANCE_C = 0.25
AT_TARGET_C = 0.1  # the device rounds to a tenth: 17.85 reads 17.8
CLOUD_GRACE = timedelta(minutes=45)  # how long a write may take to show in the cloud


@dataclass(frozen=True)
class Expectation:
    """A write that went out: what the thermostat showed before it and what it should show."""

    old: float
    target: float
    until: datetime  # the cloud may take until then to show the target
    confirmed: bool  # the target was seen right after the write


@dataclass(frozen=True)
class LKResult:
    action: str  # wrote | kept | pending | paused | failed
    target_c: float | None
    reason: str
    at: datetime


class ThermostatClient(Protocol):
    def get_climate_state(self, entity_id: str) -> dict[str, Any]: ...

    def set_temperature(self, entity_id: str, temperature: float) -> None: ...


class LKExecutor:
    """Applies the LK thermostat lever. Never raises: problems are results."""

    def __init__(
        self,
        ha: ThermostatClient,
        settings: PlanSettings,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._ha = ha
        self._s = settings
        self._now = now
        self._sleep = sleep
        self._last_written: dict[str, float] | None = None
        # values written but not seen before the wait ran out; the cloud may still deliver them
        self._pending: dict[str, float] = {}
        self._expect: dict[str, Expectation] = {}
        self._pause_until: datetime | None = None
        self.last_result: LKResult | None = None

    def use_settings(self, settings: PlanSettings) -> None:
        self._s = settings

    def export_state(self) -> dict[str, Any]:
        return {
            "last_written": self._last_written,
            "pending": self._pending,
            "expect": {
                entity: [e.old, e.target, e.until.isoformat(), e.confirmed]
                for entity, e in self._expect.items()
            },
            "pause_until": (
                None if self._pause_until is None else self._pause_until.isoformat()
            ),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        written = state["last_written"]
        self._last_written = written if isinstance(written, dict) else None
        pending = state.get("pending")  # absent in older state files
        self._pending = pending if isinstance(pending, dict) else {}
        expect = state.get("expect")  # absent in older state files
        self._expect = (
            {
                entity: Expectation(
                    old, target, datetime.fromisoformat(until), confirmed
                )
                for entity, (old, target, until, confirmed) in expect.items()
            }
            if isinstance(expect, dict)
            else {}
        )
        pause = state["pause_until"]
        self._pause_until = None if pause is None else datetime.fromisoformat(pause)

    def execute(self, holds: dict[str, float] | None = None) -> LKResult:
        now = self._now()
        current = self._read_all()
        if isinstance(current, LKResult):
            return self._record(current)
        if self._pause_until is not None and now < self._pause_until:
            return self._record(
                LKResult(
                    "paused",
                    None,
                    f"manual change, paused until {self._pause_until:%H:%M}",
                    now,
                )
            )
        for entity, wanted in list(self._pending.items()):
            if abs(current[entity] - wanted) < MANUAL_CHANGE_C:  # it got there, late
                if self._last_written is not None:
                    self._last_written[entity] = wanted
                del self._pending[entity]
        if any(self._hand_changed(entity, value) for entity, value in current.items()):
            self._pause_until = now + MANUAL_PAUSE
            self._last_written = None
            self._pending = {}
            self._expect = {}
            return self._record(
                LKResult("paused", None, "thermostat changed by hand", now)
            )
        self._pause_until = None
        targets = {
            entity: (holds or {}).get(entity, self._s.lk_limit_c) for entity in current
        }
        return self._apply(targets, current, "hold" if holds else "limit", now)

    def _hand_changed(self, entity: str, value: float) -> bool:
        """A value that nothing the executor did explains. While a write is in flight the thermostat
        may show the value from before it or the one written; anything else is a hand.
        """
        expected = self._expect.get(entity)
        if expected is not None:
            return (
                abs(value - expected.old) > AT_TARGET_C
                and abs(value - expected.target) > AT_TARGET_C
                and abs(value - self._pending.get(entity, value + 10))
                >= MANUAL_CHANGE_C
            )
        return (
            self._last_written is not None
            and abs(value - self._last_written.get(entity, value)) >= MANUAL_CHANGE_C
            and abs(value - self._pending.get(entity, value + 10)) >= MANUAL_CHANGE_C
        )

    def expect_base(self) -> None:
        """As the pump executor's: the base may arrive, and the value it replaces is still fine."""
        entities = self._s.room_thermostat_entities
        self._pending = dict.fromkeys(entities, self._s.lk_base_c)
        self._pause_until = None

    def restore_base(self) -> LKResult:
        """Thermostats back to the base (kill switch, leaving active mode, trouble)."""
        now = self._now()
        current = self._read_all()
        if isinstance(current, LKResult):
            return self._record(current)
        self._pause_until = None
        targets = dict.fromkeys(current, self._s.lk_base_c)
        return self._apply(targets, current, "restore base", now, strict=True)

    def _read_all(self) -> dict[str, float] | LKResult:
        values: dict[str, float] = {}
        for entity in self._s.room_thermostat_entities:
            try:
                values[entity] = float(
                    self._ha.get_climate_state(entity)["target_temperature"]
                )
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                return LKResult(
                    "failed", None, f"cannot read {entity}: {e}", self._now()
                )
        return values

    def _apply(
        self,
        targets: dict[str, float],
        current: dict[str, float],
        reason: str,
        now: datetime,
        strict: bool = False,
    ) -> LKResult:
        """Write what differs. `strict` (restoring the base) counts a write the cloud has not shown
        yet as a failure; otherwise it is pending until `CLOUD_GRACE` has passed."""
        failures = []
        waiting = []
        wrote = False
        done: dict[str, float] = (
            {}
        )  # thermostats now at their target, whatever happened to the others
        for entity, value in current.items():
            target = targets[entity]
            if abs(value - target) <= AT_TARGET_C:
                done[entity] = target
                self._pending.pop(entity, None)
                continue
            expected = self._expect.get(entity)
            same_write = (
                not strict
                and expected is not None
                and abs(expected.target - target) <= AT_TARGET_C
            )
            if same_write and now < expected.until:
                waiting.append(entity)  # the cloud has not shown it yet
                continue
            if same_write and not expected.confirmed:
                minutes = int(CLOUD_GRACE.total_seconds() // 60)
                failures.append(
                    f"{entity}: still {value} more than {minutes} min after writing {target}"
                )
            try:
                self._ha.set_temperature(entity, target)
                confirmed = read_until(
                    lambda entity=entity: float(
                        self._ha.get_climate_state(entity)["target_temperature"]
                    ),
                    target,
                    CONFIRM_TOLERANCE_C,
                    self._sleep,
                )
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                failures.append(f"{entity}: {e}")
                self._pending[entity] = (
                    target  # the call may have gone through all the same
                )
                continue
            seen = abs(confirmed - target) <= CONFIRM_TOLERANCE_C
            self._expect[entity] = Expectation(value, target, now + CLOUD_GRACE, seen)
            if seen:
                self._pending.pop(entity, None)
                done[entity] = target
                wrote = True
            elif strict:
                failures.append(f"{entity}: read back {confirmed}, wrote {target}")
                self._pending[entity] = target
            else:
                self._pending[entity] = target
                waiting.append(entity)
                wrote = True
        shown = (
            self._s.lk_limit_c
            if len(set(targets.values())) != 1
            else next(iter(targets.values()))
        )
        self._last_written = {**(self._last_written or {}), **done}
        if failures:
            logger.error("LK thermostat write failed: %s", "; ".join(failures))
            return self._record(LKResult("failed", shown, "; ".join(failures), now))
        if waiting:
            names = ", ".join(waiting)
            logger.info("LK thermostats waiting for the cloud: %s", names)
            return self._record(
                LKResult(
                    "pending",
                    shown,
                    f"{reason}: not shown yet by the cloud ({names})",
                    now,
                )
            )
        if wrote:
            logger.info("LK thermostats -> %s (%s)", targets, reason)
        return self._record(LKResult("wrote" if wrote else "kept", shown, reason, now))

    def _record(self, result: LKResult) -> LKResult:
        self.last_result = result
        return result
