"""REST endpoints for the salon and guest house radiator plan (shadow mode: nothing is written)."""

import asyncio
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

from core.theria.exceptions import ConfigurationError
from core.theria.radiators.service import RadiatorService

router = APIRouter(prefix="/api/radiators")

# Set by app.py during startup, like the plan service in plan_api.
service: RadiatorService | None = None


def get_service() -> RadiatorService:
    if service is None:
        raise HTTPException(
            status_code=503, detail="Radiator service is not configured"
        )
    return service


class PrefsRequest(BaseModel):
    values: dict[str, Any]


@router.get("/status")
async def get_status(svc: Annotated[RadiatorService, Depends(get_service)]) -> Any:
    """The latest shadow plan: what it would write, what the thermostats have, how readable they are."""
    snapshot = svc.latest()
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail="No radiator plan computed yet"
            + (f": {svc.last_error}" if svc.last_error else ""),
        )
    body = jsonable_encoder(snapshot)
    body["age_seconds"] = (svc.now() - snapshot.computed_at).total_seconds()
    body["availability_share"] = svc.availability_share()
    body["last_error"] = (
        svc.last_error
    )  # a failed cycle leaves the previous plan in place
    return body


@router.get("/prefs")
async def get_prefs(svc: Annotated[RadiatorService, Depends(get_service)]) -> Any:
    return jsonable_encoder(svc.prefs_catalog())


@router.put("/prefs")
async def put_prefs(
    svc: Annotated[RadiatorService, Depends(get_service)], request: PrefsRequest
) -> Any:
    try:
        await asyncio.to_thread(svc.update_prefs, request.values)
    except ConfigurationError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return jsonable_encoder(svc.prefs_catalog())


@router.delete("/prefs")
async def delete_prefs(svc: Annotated[RadiatorService, Depends(get_service)]) -> Any:
    await asyncio.to_thread(svc.reset_prefs)
    return jsonable_encoder(svc.prefs_catalog())


@router.get("/log")
async def get_log(
    svc: Annotated[RadiatorService, Depends(get_service)],
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> Any:
    return jsonable_encoder(svc.log(limit))
