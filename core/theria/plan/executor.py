"""Writing the plan's setpoint to the heat pump, with guards, and watching for trouble.

The only thing written is the room setpoint (`climate.room_temp_setpoint`). Every write is
clamped to what the current stage allows, limited when it raises the setpoint, read back, and
skipped when someone has changed the setpoint by hand. Without a fresh plan or when asked to
restore, the base setpoint is written.
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
from core.theria.plan.snapshot import PlanSnapshot
from core.theria.plan.stages import (
    StageState,
    allowed_deviation_c,
    allowed_rise_per_hour_c,
)

logger = logging.getLogger(__name__)

MAX_PLAN_AGE = timedelta(minutes=45)
MANUAL_PAUSE = timedelta(hours=6)
RAISE_INTERVAL = timedelta(minutes=55)  # one raise step per hour
MANUAL_CHANGE_C = (
    0.25  # the setpoint is this far from what was written: someone changed it
)
CONFIRM_TOLERANCE_C = 0.25
AUX_CHECKS = 2  # consecutive checks (15 min apart) with the electric heater running


@dataclass(frozen=True)
class ExecutionResult:
    action: str  # wrote | kept | fallback | paused | failed
    target_c: float | None
    written_c: float | None
    reason: str
    at: datetime


def _iso(moment: datetime | None) -> str | None:
    return None if moment is None else moment.isoformat()


def _moment(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


class SetpointClient(Protocol):
    def get_climate_state(self, entity_id: str) -> dict[str, Any]: ...

    def set_temperature(self, entity_id: str, temperature: float) -> None: ...


class PlanExecutor:
    """Applies the plan to the pump's room setpoint. Never raises: problems are results."""

    def __init__(
        self,
        ha: SetpointClient,
        settings: PlanSettings,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._ha = ha
        self._s = settings
        self._now = now
        self._sleep = sleep
        self._last_written: float | None = None
        self._last_raise_at: datetime | None = None
        self._pause_until: datetime | None = None
        self.last_result: ExecutionResult | None = None

    def export_state(self) -> dict[str, Any]:
        """What the executor remembers, so that a restart does not forget a hand change."""
        return {
            "last_written": self._last_written,
            "last_raise_at": _iso(self._last_raise_at),
            "pause_until": _iso(self._pause_until),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._last_written = state["last_written"]
        self._last_raise_at = _moment(state["last_raise_at"])
        self._pause_until = _moment(state["pause_until"])

    def execute(
        self, snapshot: PlanSnapshot | None, stage: StageState
    ) -> ExecutionResult:
        now = self._now()
        try:
            current = float(
                self._ha.get_climate_state(self._s.room_setpoint_entity)[
                    "target_temperature"
                ]
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            return self._record(
                ExecutionResult("failed", None, None, f"cannot read setpoint: {e}", now)
            )
        if self._pause_until is not None and now < self._pause_until:
            return self._record(
                ExecutionResult(
                    "paused",
                    None,
                    None,
                    f"manual change, paused until {self._pause_until:%H:%M}",
                    now,
                )
            )
        if (
            self._last_written is not None
            and abs(current - self._last_written) >= MANUAL_CHANGE_C
        ):
            self._pause_until = now + MANUAL_PAUSE
            self._last_written = None
            return self._record(
                ExecutionResult(
                    "paused", None, None, f"setpoint changed by hand to {current}", now
                )
            )
        self._pause_until = None

        deviation = allowed_deviation_c(stage)
        if snapshot is None or now - snapshot.computed_at > MAX_PLAN_AGE:
            return self._write(
                self._s.rts_base_c, current, "fallback", "no fresh plan", now, False
            )
        hour = next(
            (
                h
                for h in snapshot.plan.hours
                if h.start <= now < h.start + timedelta(hours=1)
            ),
            snapshot.plan.hours[0],
        )
        base = self._s.rts_base_c
        desired = min(max(hour.rts_c, base - deviation), base + deviation)
        desired = min(max(desired, self._s.rts_min_c), self._s.rts_max_c)
        desired = round(desired / self._s.rts_step_c) * self._s.rts_step_c
        if desired > current + 1e-9:
            if (
                self._last_raise_at is not None
                and now - self._last_raise_at < RAISE_INTERVAL
            ):
                return self._record(
                    ExecutionResult(
                        "kept", desired, None, "waiting to raise again", now
                    )
                )
            rise = allowed_rise_per_hour_c(stage)
            limit = current + rise
            if desired > limit:
                # round down to the step: rounding to the nearest could exceed the limit
                desired = (
                    math.floor(limit / self._s.rts_step_c + 1e-9) * self._s.rts_step_c
                )
        return self._write(
            desired, current, "wrote", "plan", now, desired > current + 1e-9
        )

    def restore_base(self) -> ExecutionResult:
        """Write the base setpoint (kill switch, leaving active mode)."""
        now = self._now()
        try:
            current = float(
                self._ha.get_climate_state(self._s.room_setpoint_entity)[
                    "target_temperature"
                ]
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            return self._record(
                ExecutionResult("failed", None, None, f"cannot read setpoint: {e}", now)
            )
        self._pause_until = None
        return self._write(
            self._s.rts_base_c, current, "wrote", "restore base", now, False
        )

    def _write(
        self,
        value: float,
        current: float,
        action: str,
        reason: str,
        now: datetime,
        raising: bool,
    ) -> ExecutionResult:
        if abs(value - current) < 1e-9:
            self._last_written = value  # a later hand change is then noticed
            return self._record(ExecutionResult("kept", value, None, reason, now))
        try:
            self._ha.set_temperature(self._s.room_setpoint_entity, value)
            confirmed = read_until(
                lambda: float(
                    self._ha.get_climate_state(self._s.room_setpoint_entity)[
                        "target_temperature"
                    ]
                ),
                value,
                CONFIRM_TOLERANCE_C,
                self._sleep,
            )
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            logger.error("writing the room setpoint failed: %s", e)
            return self._record(
                ExecutionResult("failed", value, None, f"write failed: {e}", now)
            )
        if abs(confirmed - value) > CONFIRM_TOLERANCE_C:
            logger.error(
                "room setpoint %s was not accepted, read back %s", value, confirmed
            )
            return self._record(
                ExecutionResult(
                    "failed", value, None, f"read back {confirmed}, wrote {value}", now
                )
            )
        self._last_written = value
        if raising:
            self._last_raise_at = now
        logger.info("room setpoint %s -> %s (%s)", current, value, reason)
        return self._record(ExecutionResult(action, value, value, reason, now))

    def _record(self, result: ExecutionResult) -> ExecutionResult:
        self.last_result = result
        return result


class StateReader(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...


class SafetyMonitor:
    """Spots trouble quickly while the plan is in control: an electric heater that runs while the
    setpoint is raised. Counts consecutive checks, so a single odd reading does not matter.
    """

    def __init__(self, states: StateReader, settings: PlanSettings):
        self._states = states
        self._s = settings
        self._aux = 0

    def check(self, raised: bool = False) -> str | None:
        """Trouble worth stepping back for, or None. A reported trouble starts the counting over."""
        aux = self._number(self._s.aux_status_entity)
        if raised and aux is not None and aux > 0.5:
            self._aux += 1
        else:
            self._aux = 0
        if self._aux < AUX_CHECKS:
            return None
        self._aux = 0
        return "the electric heater is running while the setpoint is raised"

    def _text(self, entity_id: str) -> str | None:
        try:
            return str(self._states.get_state(entity_id)["state"])
        except (ValueError, RuntimeError, KeyError):
            return None

    def _number(self, entity_id: str) -> float | None:
        try:
            return float(self._states.get_state(entity_id)["state"])
        except (ValueError, RuntimeError, KeyError, TypeError):
            return None
