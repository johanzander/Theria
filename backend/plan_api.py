"""REST endpoints for the price-driven heating plan (shadow and active mode)."""

import asyncio
from datetime import date
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from core.theria.exceptions import (
    ConfigurationError,
    SetpointRestoreError,
    TheriaError,
)
from core.theria.plan.service import PlanService
from core.theria.plan.snapshot import PlanSnapshot

router = APIRouter(prefix="/api")

# Set by app.py during startup, like the other services in api.py.
service: PlanService | None = None


def get_service() -> PlanService:
    if service is None:
        raise HTTPException(status_code=503, detail="Plan service is not configured")
    return service


class ModeRequest(BaseModel):
    mode: str
    confirm: bool = False  # required for active mode, which writes to the heat pump


def _summary(snapshot: PlanSnapshot) -> dict:
    first = snapshot.plan.hours[0]
    return {
        "computed_at": snapshot.computed_at,
        "mode": snapshot.mode,
        "rts_now_c": first.rts_c,
        "objective_sek": snapshot.plan.objective_sek,
        "baseline_objective_sek": snapshot.plan.baseline_objective_sek,
        "floor_violation_hours": snapshot.plan.floor_violation_hours,
    }


@router.get("/plan")
async def get_plan(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Latest plan with predictions, baseline and per-hour reasons."""
    snapshot = svc.latest()
    if snapshot is None:
        raise HTTPException(
            status_code=404, detail=f"No plan computed yet (mode: {svc.mode})"
        )
    body = jsonable_encoder(snapshot)
    body["age_seconds"] = (svc.now() - snapshot.computed_at).total_seconds()
    return body


@router.get("/plan/day")
async def get_plan_day(
    svc: Annotated[PlanService, Depends(get_service)],
    day: Annotated[date | None, Query()] = None,
) -> Any:
    """A whole local day (default today): measured so far, planned for the rest."""
    if svc.latest() is None:
        raise HTTPException(
            status_code=404, detail=f"No plan computed yet (mode: {svc.mode})"
        )
    chosen = day or svc.now().astimezone(ZoneInfo(svc.timezone)).date()
    return jsonable_encoder(await asyncio.to_thread(svc.day_view, chosen))


@router.get("/plan/tuning")
async def get_tuning(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Advice on the settings the plan never changes by itself; nothing is applied."""
    try:
        return jsonable_encoder(await asyncio.to_thread(svc.tuning))
    except TheriaError as e:
        raise HTTPException(status_code=422, detail=f"{type(e).__name__}: {e}") from e


@router.get("/plan/history")
async def get_plan_history(
    svc: Annotated[PlanService, Depends(get_service)],
    hours: Annotated[int, Query(ge=1, le=48)] = 24,
) -> Any:
    """Compact summaries of recent plans."""
    return jsonable_encoder([_summary(s) for s in svc.history(hours)])


@router.get("/model/health")
async def get_model_health(
    svc: Annotated[PlanService, Depends(get_service)],
) -> Any:
    """Model parameters and whether they are priors or fitted."""
    return jsonable_encoder(
        {"mode": svc.mode, "models": svc.model_status(), "scores": svc.scores()}
    )


@router.post("/plan/mode")
async def set_plan_mode(
    svc: Annotated[PlanService, Depends(get_service)], request: ModeRequest
) -> dict[str, str]:
    """Switch between off, shadow and active (active needs confirm=true)."""
    if request.mode == "active" and svc.mode != "active":
        failing = [
            c for c in await asyncio.to_thread(svc.preflight) if c.status == "fail"
        ]
        if failing:
            detail = "; ".join(f"{c.name}: {c.detail}" for c in failing)
            raise HTTPException(status_code=422, detail=f"preflight failed: {detail}")
    try:
        await asyncio.to_thread(svc.set_mode, request.mode, request.confirm)
    except ConfigurationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except SetpointRestoreError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"mode": svc.mode}


@router.post("/model/refit")
async def refit_models(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Fit the pump and room models to history now (takes several seconds)."""
    try:
        models = await asyncio.to_thread(svc.refit)
    except ConfigurationError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except TheriaError as e:
        raise HTTPException(
            status_code=422, detail=f"{type(e).__name__}: {e}; previous models kept"
        ) from e
    return jsonable_encoder(models)


@router.get("/plan/execution")
async def get_execution(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Active-mode state: last write, last trouble."""
    return jsonable_encoder(svc.execution_status())


@router.get("/savings")
async def get_savings(
    svc: Annotated[PlanService, Depends(get_service)],
    days: Annotated[int, Query(ge=1, le=90)] = 14,
) -> Any:
    """Measured heating cost per day against the baseline the pump would have had."""
    try:
        body = await asyncio.to_thread(svc.savings, days)
    except ConfigurationError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return jsonable_encoder(body)


@router.get("/plan/preflight")
async def get_preflight(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Checks to read before activating: models, plan, automation, setpoints, compressor."""
    checks = await asyncio.to_thread(svc.preflight)
    return jsonable_encoder(
        {"ok": all(c.status != "fail" for c in checks), "checks": checks}
    )


@router.get("/plan/log")
async def get_action_log(svc: Annotated[PlanService, Depends(get_service)]) -> Any:
    """Setpoint writes, trouble and mode changes, newest last."""
    return jsonable_encoder(svc.action_log())


@router.get("/plan/parameters")
async def get_parameter_history(
    svc: Annotated[PlanService, Depends(get_service)],
    limit: Annotated[int, Query(ge=1, le=5000)] = 500,
) -> Any:
    """What the plan has learned over time: every refit and every margin change."""
    return jsonable_encoder(
        {"current": svc.parameters(), "history": svc.parameter_history(limit)}
    )
