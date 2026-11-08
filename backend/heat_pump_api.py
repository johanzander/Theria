"""REST endpoint for the heat pump timeline (the Värmepump tab)."""

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException

from core.theria.exceptions import SensorError
from core.theria.heat_pump_timeline import MAX_RANGE, build_timeline

router = APIRouter(prefix="/api")


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
