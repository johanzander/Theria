"""Scoring of the room temperature predictions against what actually happened."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.theria.plan.snapshot import PlanSnapshot

LEADS_H = (1, 6, 24)
LEAD_TOLERANCE_H = {1: 0.5, 6: 0.5, 24: 1.0}
READING_TOLERANCE = timedelta(minutes=10)


@dataclass(frozen=True)
class LeadScore:
    """Prediction error of the room temperature `lead_h` hours ahead (C)."""

    lead_h: int
    n: int
    mae_c: float
    bias_c: float  # predicted minus actual


class ScoreBoard:
    """Collects measured room temperatures and scores snapshots against them."""

    def __init__(self, max_readings: int = 96 * 7):
        self._readings: deque[tuple[datetime, float]] = deque(maxlen=max_readings)

    def record(self, moment: datetime, room_c: float) -> None:
        self._readings.append((moment, room_c))

    def scores(self, snapshots: Sequence[PlanSnapshot]) -> list[LeadScore]:
        errors: dict[int, list[float]] = {lead: [] for lead in LEADS_H}
        readings = list(self._readings)
        for snapshot in snapshots:
            for hour in snapshot.plan.hours:
                end = hour.start + timedelta(hours=1)
                lead_h = (end - snapshot.computed_at) / timedelta(hours=1)
                lead = _bucket(lead_h)
                if lead is None:
                    continue
                actual = _reading_near(readings, end)
                if actual is not None:
                    errors[lead].append(hour.room_c_end + hour.sun_swing_c - actual)
        return [
            LeadScore(
                lead_h=lead,
                n=len(values),
                mae_c=float(np.mean(np.abs(values))),
                bias_c=float(np.mean(values)),
            )
            for lead, values in errors.items()
            if values
        ]


def _bucket(lead_h: float) -> int | None:
    for lead in LEADS_H:
        if abs(lead_h - lead) <= LEAD_TOLERANCE_H[lead]:
            return lead
    return None


def _reading_near(
    readings: list[tuple[datetime, float]], moment: datetime
) -> float | None:
    close = [
        (abs(t - moment), value)
        for t, value in readings
        if abs(t - moment) <= READING_TOLERANCE
    ]
    return min(close)[1] if close else None
