"""Writing the LK room thermostat setpoints, the second lever next to the pump setpoint.

The LK valves close when the room reaches the thermostat setpoint, so with the thermostats at
the base the floor cannot store heat above it. While the pump setpoint is actually above its
base (pre-heating) the thermostats are raised to `lk_setpoint_c`; otherwise they are at
`lk_base_c`. They are never set below the base, so the loops are never closed more than today.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from core.theria.plan.settings import PlanSettings

logger = logging.getLogger(__name__)

MANUAL_PAUSE = timedelta(hours=6)
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
    ):
        self._ha = ha
        self._s = settings
        self._now = now
        self._last_written: float | None = None
        self._pause_until: datetime | None = None
        self.last_result: LKResult | None = None

    def execute(self, pump_setpoint_c: float) -> LKResult:
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
            abs(v - self._last_written) >= MANUAL_CHANGE_C for v in current.values()
        ):
            self._pause_until = now + MANUAL_PAUSE
            self._last_written = None
            return self._record(
                LKResult("paused", None, "thermostat changed by hand", now)
            )
        self._pause_until = None
        preheating = pump_setpoint_c > self._s.rts_base_c + 1e-9
        target = self._s.lk_setpoint_c if preheating else self._s.lk_base_c
        return self._apply(
            target, current, "pre-heating" if preheating else "base", now
        )

    def restore_base(self) -> LKResult:
        """Thermostats back to the base (kill switch, leaving active mode, trouble)."""
        now = self._now()
        current = self._read_all()
        if isinstance(current, LKResult):
            return self._record(current)
        self._pause_until = None
        return self._apply(self._s.lk_base_c, current, "restore base", now)

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
        self, target: float, current: dict[str, float], reason: str, now: datetime
    ) -> LKResult:
        failures = []
        lowering = target <= self._s.lk_base_c + 1e-9
        for entity, value in current.items():
            if abs(value - target) < 1e-9 or (lowering and value < target):
                continue  # right already, or set below the base by hand: not ours to raise
            try:
                self._ha.set_temperature(entity, target)
                confirmed = float(
                    self._ha.get_climate_state(entity)["target_temperature"]
                )
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                failures.append(f"{entity}: {e}")
                continue
            if abs(confirmed - target) > CONFIRM_TOLERANCE_C:
                failures.append(f"{entity}: read back {confirmed}, wrote {target}")
        if failures:
            logger.error("LK thermostat write failed: %s", "; ".join(failures))
            return self._record(LKResult("failed", target, "; ".join(failures), now))
        wrote = any(
            abs(v - target) >= 1e-9 and not (lowering and v < target)
            for v in current.values()
        )
        if wrote:
            self._last_written = target
            logger.info("LK thermostats -> %s (%s)", target, reason)
        else:
            self._last_written = target
        return self._record(LKResult("wrote" if wrote else "kept", target, reason, now))

    def _record(self, result: LKResult) -> LKResult:
        self.last_result = result
        return result
