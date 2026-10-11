"""One computed plan with the context it was computed in."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.theria.plan.planner import Plan
from core.theria.plan.room_model import RoomState


@dataclass(frozen=True)
class ComfortOutlook:
    """Where the room is expected to be when comfort starts, and how long a deficit would take."""

    start: datetime  # when the night ends
    room_c: float  # planned room temperature then
    target_c: float
    deficit_c: float  # how far under the target (0 when it is there)
    extra_rate_c_per_h: (
        float  # how much faster the room can be warmed than the plan does
    )
    lead_hours: (
        float | None
    )  # hours of maximum warming the deficit would take; None: cannot


@dataclass(frozen=True)
class PlanSnapshot:
    computed_at: datetime
    mode: str
    plan: Plan
    extreme_threshold_sek_per_kwh: float
    outdoor_bias_c: float
    initial: RoomState
    model_status: dict[str, Any]  # which model parameters are priors vs fitted
    comfort_outlook: ComfortOutlook | None = None
