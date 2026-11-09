"""REST endpoints for the heat pump: its history (Värmepump tab) and its readings now (Overview)."""

import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from core.theria.exceptions import SensorError
from core.theria.ha_client import HAClient
from core.theria.heat_pump_timeline import MAX_RANGE, build_timeline, read_pump_now

router = APIRouter(prefix="/api")

# Set by app.py during startup, like the other services.
ha_client: HAClient | None = None


def get_ha_client() -> HAClient:
    if ha_client is None:
        raise HTTPException(status_code=503, detail="Home Assistant is not configured")
    return ha_client


@router.get("/heatpump/now")
async def get_pump_now(client: Annotated[HAClient, Depends(get_ha_client)]) -> Any:
    """What the heat pump's sensors in Home Assistant read right now."""
    return await asyncio.to_thread(read_pump_now, client)


@router.get("/heatpump/timeline")
async def get_timeline(start: datetime, stop: datetime) -> Any:
    """Everything the heat pump did in the range, on one time axis. A stop in the future means now."""
    if start.tzinfo is None or stop.tzinfo is None:
        raise HTTPException(status_code=422, detail="start and stop need a timezone")
    stop = min(stop, datetime.now(UTC))
    if stop <= start or stop - start > MAX_RANGE:
        raise HTTPException(
            status_code=422,
            detail=f"the range must be positive and at most {MAX_RANGE.days} days",
        )
    try:
        return await asyncio.to_thread(build_timeline, start, stop)
    except SensorError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
