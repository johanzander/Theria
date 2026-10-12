"""A fast correction from how the room really moved against what the last plan expected.

Each cycle compares the room's usual level now with what an earlier plan (a few hours ago)
predicted for this moment. The shortfall per hour is the model's drift error; it is smoothed
into an estimate that the next plan adds to its drift for the coming hours and that fades with
time, because what the house does that the model cannot predict (stove, sun, people) does not
last. Nothing is persisted: after a restart the estimate starts from zero.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

import numpy as np

from core.theria.plan.planner import HourPlan

GAIN = 0.3  # share of a new error rate taken into the estimate per update
HALF_LIFE_H = 4.0  # the estimate (and its reach into the future) halves in this time
MAX_DRIFT_C_PER_H = 0.3  # sanity limit on what the plan may be told


class DriftObserver:
    def __init__(self) -> None:
        self.drift_c_per_h = 0.0
        self._at: datetime | None = None

    def at(self, moment: datetime) -> float:
        """The estimate faded to `moment`."""
        if self._at is None:
            return 0.0
        hours = (moment - self._at).total_seconds() / 3600
        return self.drift_c_per_h * 0.5 ** (max(hours, 0.0) / HALF_LIFE_H)

    def update(
        self, now: datetime, measured_c: float, predicted_c: float, lead_h: float
    ) -> None:
        """Take in how far the room is from what the plan `lead_h` hours ago expected for now."""
        rate = (measured_c - predicted_c) / lead_h
        faded = self.at(now)
        estimate = faded + GAIN * (rate - faded)
        self.drift_c_per_h = float(
            np.clip(estimate, -MAX_DRIFT_C_PER_H, MAX_DRIFT_C_PER_H)
        )
        self._at = now

    def correction(self, hour_starts: Sequence[datetime], now: datetime) -> np.ndarray:
        """Drift to add per planned hour (the further ahead, the less)."""
        return np.array([self.at(max(h, now)) for h in hour_starts], dtype=float)


def predicted_room_c(
    hours: Sequence[HourPlan], initial_room_c: float, moment: datetime
) -> float | None:
    """The room temperature a plan expected at `moment` (linear within the hour)."""
    for i, hour in enumerate(hours):
        if hour.start <= moment < hour.start + timedelta(hours=1):
            before = initial_room_c if i == 0 else hours[i - 1].room_c_end
            share = (moment - hour.start) / timedelta(hours=1)
            return before + share * (hour.room_c_end - before)
    return None
