"""Writing the LK room thermostat setpoints, room by room.

The LK valves close when the room reaches the thermostat setpoint. In the heat shift each room gets
its target for the hour (`holds`): closing in a sell, above the room in a buy just before a sell, and
the normal target otherwise. In the optimizer the setpoint is the room limit (`lk_limit_c`), and the
regulator may hold a single thermostat lower for a while. A room changed by hand is paused on its own.
"""

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from core.theria.plan.confirm import read_until
from core.theria.plan.settings import PlanSettings

logger = logging.getLogger(__name__)

MANUAL_PAUSE = timedelta(hours=1)
PENDING_GRACE = timedelta(hours=1)  # a late cloud confirmation counts as ours this long
MANUAL_CHANGE_C = 0.25
CONFIRM_TOLERANCE_C = 0.25
AT_TARGET_C = (
    0.1  # the thermostats show their value to a tenth: a value that close is the target
)
LK_STEP_C = (
    0.5  # the thermostats' target_temp_step: other values end up as 17.8 or 17.7
)


def step_up(value_c: float) -> float:
    """The lowest valid thermostat setpoint that is not below `value_c`."""
    return math.ceil(value_c / LK_STEP_C - 1e-9) * LK_STEP_C


@dataclass(frozen=True)
class LKResult:
    action: str  # wrote | kept | paused | failed
    target_c: float | None
    reason: str
    at: datetime


class ThermostatClient(Protocol):
    def get_climate_state(self, entity_id: str) -> dict[str, Any]: ...

    def set_temperature(self, entity_id: str, temperature: float) -> None: ...


class LKExecutor:
    """Applies the LK thermostat lever, room by room. Never raises: problems are results."""

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
        self._last_written: dict[str, float] = {}
        self._known = False  # something has been written or read as ours
        # values written but not seen before the wait ran out; the cloud may still deliver them
        self._pending: dict[str, list[tuple[float, datetime]]] = {}
        self._pause_until: dict[str, datetime] = {}
        # heat shift: rooms changed by hand, and the window that was going on (None: none was)
        self._left_alone: dict[str, str | None] = {}
        self.last_result: LKResult | None = None
        # what each thermostat showed before the last cycle and what the cycle asked of it
        self.last_read: dict[str, float] | None = None
        self.last_targets: dict[str, float] | None = None

    def use_settings(self, settings: PlanSettings) -> None:
        self._s = settings

    def export_state(self) -> dict[str, Any]:
        return {
            "last_written": self._last_written if self._known else None,
            "pending": {
                entity: [[value, at.isoformat()] for value, at in values]
                for entity, values in self._pending.items()
            },
            "paused": {e: at.isoformat() for e, at in self._pause_until.items()},
            "left_alone": self._left_alone,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        written = state["last_written"]
        self._known = isinstance(written, dict)
        self._last_written = written if isinstance(written, dict) else {}
        now = self._now()
        self._pending = {}
        pending = state.get("pending")  # absent in older state files
        for entity, values in (pending if isinstance(pending, dict) else {}).items():
            # older files keep one value and no time
            self._pending[entity] = (
                [(float(v), datetime.fromisoformat(at)) for v, at in values]
                if isinstance(values, list)
                else [(float(values), now)]
            )
        self._pause_until = {
            e: datetime.fromisoformat(at) for e, at in state.get("paused", {}).items()
        }
        self._left_alone = dict(state.get("left_alone", {}))

    def _read_one(self, entity: str) -> float | str:
        try:
            return float(self._ha.get_climate_state(entity)["target_temperature"])
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            return f"cannot read {entity}: {e}"

    def _expire_pending(self, now: datetime) -> None:
        for entity, values in list(self._pending.items()):
            kept = [(v, at) for v, at in values if now - at <= PENDING_GRACE]
            if kept:
                self._pending[entity] = kept
            else:
                del self._pending[entity]

    def _seen(self, entity: str, value: float) -> bool:
        """Whether `value` is one Theria asked for and the cloud delivered late."""
        values = self._pending.get(entity, [])
        match = next(
            (i for i, (v, _) in enumerate(values) if abs(value - v) < MANUAL_CHANGE_C),
            None,
        )
        if match is None:
            return False
        self._last_written[entity] = value
        remaining = values[match + 1 :]
        if remaining:
            self._pending[entity] = remaining
        else:
            del self._pending[entity]
        return True

    def _by_hand(self, entity: str, value: float) -> bool:
        return (
            entity in self._last_written
            and abs(value - self._last_written[entity]) >= MANUAL_CHANGE_C
        )

    def _skip(self, entity: str, window: str | None, now: datetime) -> bool:
        """Whether a room is left alone this cycle."""
        if entity in self._pause_until and now < self._pause_until[entity]:
            return True
        self._pause_until.pop(entity, None)
        if entity not in self._left_alone:
            return False
        if window is None or window == self._left_alone[entity]:
            return True  # not before the next window
        del self._left_alone[entity]
        return False

    def execute(
        self, holds: dict[str, float] | None = None, window: str | None = None
    ) -> LKResult:
        now = self._now()
        self.last_read = self.last_targets = None
        self._expire_pending(now)
        current: dict[str, float] = {}
        failures: list[str] = []
        for entity in self._s.room_thermostat_entities:
            read = self._read_one(entity)
            if isinstance(read, str):
                failures.append(read)
            else:
                current[entity] = read
        self.last_read = dict(current)
        changed: list[str] = []
        for entity, value in current.items():
            if self._seen(entity, value) or not self._by_hand(entity, value):
                continue
            changed.append(entity)
            self._pause_until[entity] = now + MANUAL_PAUSE
            self._last_written.pop(entity, None)
            self._pending.pop(entity, None)
            if self._s.plan_strategy == "heat_shift":
                self._left_alone[entity] = window
        free = {e: v for e, v in current.items() if not self._skip(e, window, now)}
        default = self._s.lk_setpoint_c
        targets = {e: (holds or {}).get(e, default) for e in free}
        self.last_targets = targets if targets else None
        result = self._apply(targets, free, "hold" if holds else "limit", now, failures)
        if result.action == "failed":
            return self._record(result)
        if changed:
            return self._record(
                LKResult("paused", None, f"changed by hand: {', '.join(changed)}", now)
            )
        paused = [e for e in current if e in self._pause_until]
        if paused:
            until = min(self._pause_until[e] for e in paused)
            return self._record(
                LKResult(
                    "paused",
                    None,
                    f"manual change, {', '.join(paused)} paused until {until:%H:%M}",
                    now,
                )
            )
        return self._record(result)

    def expect_base(self) -> None:
        """As the pump executor's: the base may arrive, and the value it replaces is still fine."""
        now = self._now()
        for entity in self._s.room_thermostat_entities:
            self._pending.setdefault(entity, []).append((self._s.lk_base_c, now))
        self._pause_until = {}
        self._left_alone = {}

    def restore_base(self) -> LKResult:
        """Thermostats back to the base (kill switch, leaving active mode, trouble)."""
        now = self._now()
        current: dict[str, float] = {}
        failures: list[str] = []
        for entity in self._s.room_thermostat_entities:
            read = self._read_one(entity)
            if isinstance(read, str):
                failures.append(read)
            else:
                current[entity] = read
        self._pause_until = {}
        self._left_alone = {}
        targets = dict.fromkeys(current, self._s.lk_base_c)
        return self._record(
            self._apply(targets, current, "restore base", now, failures)
        )

    def _apply(
        self,
        targets: dict[str, float],
        current: dict[str, float],
        reason: str,
        now: datetime,
        failures: list[str],
    ) -> LKResult:
        wrote = False
        for entity, value in current.items():
            target = targets[entity]
            if abs(value - target) <= AT_TARGET_C:
                self._last_written[entity] = target
                continue
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
                self._pending.setdefault(entity, []).append((target, now))
                continue  # the call may have gone through all the same
            if abs(confirmed - target) > CONFIRM_TOLERANCE_C:
                failures.append(f"{entity}: read back {confirmed}, wrote {target}")
                self._pending.setdefault(entity, []).append((target, now))
                continue
            self._pending.pop(entity, None)
            self._last_written[entity] = target
            wrote = True
        self._known = self._known or bool(self._last_written)
        shown = (
            next(iter(targets.values()))
            if len(set(targets.values())) == 1
            else self._s.lk_setpoint_c
        )
        if failures:
            logger.error("LK thermostat write failed: %s", "; ".join(failures))
            return LKResult("failed", shown, "; ".join(failures), now)
        if wrote:
            logger.info("LK thermostats -> %s (%s)", targets, reason)
        return LKResult("wrote" if wrote else "kept", shown, reason, now)

    def _record(self, result: LKResult) -> LKResult:
        self.last_result = result
        return result
