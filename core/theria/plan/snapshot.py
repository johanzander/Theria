"""One computed plan with the context it was computed in."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.theria.plan.planner import Plan
from core.theria.plan.room_model import RoomState


@dataclass(frozen=True)
class PlanSnapshot:
    computed_at: datetime
    mode: str
    plan: Plan
    extreme_threshold_sek_per_kwh: float
    outdoor_bias_c: float
    initial: RoomState
    model_status: dict[str, Any]  # which model parameters are priors vs fitted
