"""Writing the LK room thermostat setpoints: a static limit per room.

The LK valves close when the room reaches the thermostat setpoint, so the setpoint is the room
limit (`lk_limit_c`): the floor can store heat up to it and no further. It is held there at all
times. The regulator may hold a single thermostat lower to close that room's valves for a while
(`holds`); the thermostat goes back to the limit when the hold is released.
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
        self._pause_until: datetime | None = None
        self.last_result: LKResult | None = None

    def export_state(self) -> dict[str, Any]:
        return {
            "last_written": self._last_written,
            "pause_until": (
                None if self._pause_until is None else self._pause_until.isoformat()
            ),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        written = state["last_written"]
        self._last_written = written if isinstance(written, dict) else None
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
        if self._last_written is not None and any(
            abs(value - self._last_written.get(entity, value)) >= MANUAL_CHANGE_C
            for entity, value in current.items()
        ):
            self._pause_until = now + MANUAL_PAUSE
            self._last_written = None
            return self._record(
                LKResult("paused", None, "thermostat changed by hand", now)
            )
        self._pause_until = None
        targets = {
            entity: (holds or {}).get(entity, self._s.lk_limit_c) for entity in current
        }
        return self._apply(targets, current, "hold" if holds else "limit", now)

    def expect_base(self) -> None:
        """As the pump executor's: the base is what the thermostats will be left at."""
        entities = self._s.room_thermostat_entities
        self._last_written = dict.fromkeys(entities, self._s.lk_base_c)
        self._pause_until = None

    def restore_base(self) -> LKResult:
        """Thermostats back to the base (kill switch, leaving active mode, trouble)."""
        now = self._now()
        current = self._read_all()
        if isinstance(current, LKResult):
            return self._record(current)
        self._pause_until = None
        targets = dict.fromkeys(current, self._s.lk_base_c)
        return self._apply(targets, current, "restore base", now)

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
    ) -> LKResult:
        failures = []
        wrote = False
        for entity, value in current.items():
            target = targets[entity]
            if abs(value - target) < 1e-9:
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
                continue
            if abs(confirmed - target) > CONFIRM_TOLERANCE_C:
                failures.append(f"{entity}: read back {confirmed}, wrote {target}")
                continue
            wrote = True
        shown = self._s.lk_limit_c if len(set(targets.values())) != 1 else next(iter(targets.values()))
        if failures:
            logger.error("LK thermostat write failed: %s", "; ".join(failures))
            return self._record(LKResult("failed", shown, "; ".join(failures), now))
        self._last_written = dict(targets)
        if wrote:
            logger.info("LK thermostats -> %s (%s)", targets, reason)
        return self._record(LKResult("wrote" if wrote else "kept", shown, reason, now))

    def _record(self, result: LKResult) -> LKResult:
        self.last_result = result
        return result
