"""What the room does that the model does not explain, estimated from measured inputs every cycle.

Each cycle the room is advanced one step from the previous sample with the model, using what was
actually measured in between: the flow that reached the floor, how many loops were open, the
outdoor temperature and the sun. The difference to the measured temperature, per hour, is the
innovation; it is smoothed into one slowly varying disturbance that the next plan adds to its drift
for the coming hours and that fades with time, because what the house does that the model cannot
predict (stove, people, a window) does not last. Because the inputs are the measured ones, the plan's
own effect (the flow it chose, valves that opened or closed) is not read as disturbance.

While the sun is not in the model (`k_sun == 0`) sunny samples are skipped: the estimator learns in
the hours without sun. Nothing is persisted: after a restart the estimate starts from zero.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from core.theria.plan.room_model import RoomParams, step

GAIN = 0.3  # share of a new error rate taken into the estimate per update
HALF_LIFE_H = 4.0  # the estimate (and its reach into the future) halves in this time
MAX_DRIFT_C_PER_H = 0.3  # sanity limit on what the plan may be told
MAX_GAP = timedelta(hours=1)  # a longer silence is not judged
MIN_STEP_H = 5 / 60  # shorter steps are too noisy to judge; they accumulate
SUN_OFF_WM2 = 20.0
SLAB_INIT_FRACTION = 0.5  # the slab sits half way between the room and the flow, as in the plan


@dataclass(frozen=True)
class Sample:
    at: datetime
    room_c: float
    flow_c: float  # what reaches the floor (the pump's flow up to the shunts' cap)
    valve: float | None  # measured share of open loops (0-1); None: use the model's valve
    t_out_c: float
    sun_wm2: float


class DisturbanceEstimator:
    def __init__(self) -> None:
        self.drift_c_per_h = 0.0
        self._at: datetime | None = None
        self._last: Sample | None = None
        self._slab_c = 0.0
        self._sun_filtered = 0.0

    def at(self, moment: datetime) -> float:
        """The estimate faded to `moment`."""
        if self._at is None:
            return 0.0
        hours = (moment - self._at).total_seconds() / 3600
        return self.drift_c_per_h * 0.5 ** (max(hours, 0.0) / HALF_LIFE_H)

    def update(self, sample: Sample, params: RoomParams, lk_limit_c: float) -> None:
        previous = self._last
        if previous is None or sample.at - previous.at > MAX_GAP:
            self._start(sample)
            return
        hours = (sample.at - previous.at).total_seconds() / 3600
        if hours < MIN_STEP_H:
            return
        if previous.valve is None:
            setpoint, scale = lk_limit_c, 1.0
        else:  # the measured valves replace the model's: it sees them as fully open
            setpoint, scale = previous.room_c + 10.0, previous.valve
        predicted, slab, sun_filtered = step(
            params,
            previous.room_c,
            self._slab_c,
            self._sun_filtered,
            sample.flow_c,
            sample.t_out_c,
            sample.sun_wm2,
            setpoint,
            dt_h=hours,
            valve_scale=scale,
        )
        sunny = max(previous.sun_wm2, sample.sun_wm2) > SUN_OFF_WM2
        if params.k_sun > 0 or not sunny:
            rate = (sample.room_c - float(predicted)) / hours
            faded = self.at(sample.at)
            estimate = faded + GAIN * (rate - faded)
            self.drift_c_per_h = float(
                np.clip(estimate, -MAX_DRIFT_C_PER_H, MAX_DRIFT_C_PER_H)
            )
            self._at = sample.at
        self._slab_c, self._sun_filtered = float(slab), float(sun_filtered)
        self._last = sample

    def _start(self, sample: Sample) -> None:
        self._last = sample
        self._slab_c = sample.room_c + SLAB_INIT_FRACTION * (
            sample.flow_c - sample.room_c
        )
        self._sun_filtered = sample.sun_wm2

    def correction(self, hour_starts: Sequence[datetime], now: datetime) -> np.ndarray:
        """Drift to add per planned hour (the further ahead, the less)."""
        return np.array([self.at(max(h, now)) for h in hour_starts], dtype=float)
