"""REST endpoints for the comfort sensor guard."""

import asyncio
from dataclasses import asdict
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.encoders import jsonable_encoder

from core.theria.exceptions import SensorError
from core.theria.sensor_guard import STATUS_OK, SensorGuard, SensorReport

router = APIRouter(prefix="/api")

# Set by app.py during startup, like the other services.
guard: SensorGuard | None = None


def get_guard() -> SensorGuard:
    if guard is None:
        raise HTTPException(status_code=503, detail="Sensor guard is not configured")
    return guard


_RANK = {"ok": 0, "warning": 1, "fault": 2}


def _overall(reports: list[SensorReport]) -> str:
    if not reports:
        return "unknown"
    return max((r.status for r in reports), key=lambda s: _RANK[s])


def _body(g: SensorGuard) -> dict[str, Any]:
    reports = g.reports()
    room_c: float | None = None
    note = ""
    if reports:
        try:
            room_c = g.current_room_temperature()
        except SensorError as e:
            note = str(e)
    body: dict[str, Any] = jsonable_encoder(
        {
            "status": _overall(reports),
            "checked_at": g.checked_at(),
            "sensors": [asdict(r) for r in reports],
            "room_temperature_c": room_c,
            "room_temperature_note": note,
        }
    )
    return body


@router.get("/comfort-sensors")
async def get_comfort_sensors(g: Annotated[SensorGuard, Depends(get_guard)]) -> Any:
    """Status of every comfort sensor and the room temperature the plan would use."""
    return _body(g)


@router.post("/comfort-sensors/recheck")
async def recheck_comfort_sensors(g: Annotated[SensorGuard, Depends(get_guard)]) -> Any:
    """Check the sensors now."""
    await asyncio.to_thread(g.check)
    return _body(g)


__all__ = ["STATUS_OK", "router"]
