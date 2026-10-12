"""Checks to read before letting the plan write to the heat pump."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from core.theria.plan.settings import PlanSettings
from core.theria.plan.snapshot import PlanSnapshot
from core.theria.plan.stages import StageState

MAX_PLAN_AGE = timedelta(minutes=30)
READ_ERRORS = (RuntimeError, ValueError, KeyError, TypeError)


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str


class StateClient(Protocol):
    def get_state(self, entity_id: str) -> dict[str, Any]: ...

    def get_climate_state(self, entity_id: str) -> dict[str, Any]: ...


def preflight_checks(
    ha: StateClient,
    settings: PlanSettings,
    snapshot: PlanSnapshot | None,
    pump_fitted: bool,
    room_fitted: bool,
    stage: StageState,
    now: datetime,
) -> list[Check]:
    s = settings
    checks = [
        _models(pump_fitted, room_fitted),
        _plan(snapshot, now),
        _automation(ha, s),
        _pump_setpoint(ha, s),
        _thermostats(ha, s),
        _outdoor(ha, s),
        _stage(stage),
    ]
    if snapshot is not None and snapshot.plan.floor_violation_hours > 0:
        checks.append(
            Check(
                "plan comfort",
                "warn",
                f"the plan goes below the floor in {snapshot.plan.floor_violation_hours} h",
            )
        )
    else:
        checks.append(
            Check("plan comfort", "ok", "the plan keeps the room above the floor")
        )
    return checks


def _models(pump_fitted: bool, room_fitted: bool) -> Check:
    if not pump_fitted:
        return Check("models", "fail", "the pump model is still the prior; refit first")
    if not room_fitted:
        return Check("models", "warn", "the room model is still the prior")
    return Check("models", "ok", "pump and room models are fitted")


def _plan(snapshot: PlanSnapshot | None, now: datetime) -> Check:
    if snapshot is None:
        return Check("plan", "fail", "no plan has been computed")
    age = now - snapshot.computed_at
    if age > MAX_PLAN_AGE:
        return Check(
            "plan", "fail", f"the plan is {int(age.total_seconds() // 60)} min old"
        )
    return Check("plan", "ok", f"computed {int(age.total_seconds() // 60)} min ago")


def _automation(ha: StateClient, s: PlanSettings) -> Check:
    entity = s.conflicting_automation_entity
    try:
        state = str(ha.get_state(entity)["state"])
    except READ_ERRORS as e:
        return Check("automation", "fail", f"cannot read {entity}: {e}")
    if state == "on":
        return Check("automation", "fail", f"{entity} is on and would fight the plan")
    return Check("automation", "ok", f"{entity} is {state}")


def _pump_setpoint(ha: StateClient, s: PlanSettings) -> Check:
    try:
        value = float(
            ha.get_climate_state(s.room_setpoint_entity)["target_temperature"]
        )
    except READ_ERRORS as e:
        return Check("pump setpoint", "fail", f"cannot read the setpoint: {e}")
    if not s.rts_min_c <= value <= s.rts_max_c:
        return Check(
            "pump setpoint",
            "fail",
            f"{value} C is outside {s.rts_min_c}-{s.rts_max_c} C; set the base first",
        )
    return Check("pump setpoint", "ok", f"{value} C")


def _thermostats(ha: StateClient, s: PlanSettings) -> Check:
    values = []
    for entity in s.room_thermostat_entities:
        try:
            values.append(float(ha.get_climate_state(entity)["target_temperature"]))
        except READ_ERRORS as e:
            return Check("LK thermostats", "fail", f"cannot read {entity}: {e}")
    return Check("LK thermostats", "ok", ", ".join(f"{v} C" for v in values))


def _outdoor(ha: StateClient, s: PlanSettings) -> Check:
    try:
        value = float(ha.get_state(s.outdoor_entity)["state"])
    except READ_ERRORS as e:
        return Check(
            "outdoor sensor", "fail", f"cannot read the outdoor temperature: {e}"
        )
    return Check("outdoor sensor", "ok", f"{value} C")


def _stage(stage: StageState) -> Check:
    return Check("stage", "ok", f"stage {stage.index + 1}")
