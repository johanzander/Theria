"""How much one degree of flow temperature moves the room, and how fast (daily means).

The ground-floor room is slow (days) and is also moved by things the data does not show
(stove, people, doors), so hour-by-hour prediction is not possible. What the planner needs
is the *response*: per day, `R' = k_out (T - R) + k_flow (F - R) + k_sun * sun + b`. The
fit is made season by season; it is accepted when the response to the flow temperature is
there in every season and the seasons agree about it. Its predictions are NOT better than
"no change" and are reported only for information.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

from core.theria.exceptions import ModelFitError
from core.theria.plan.room_model import RoomParams

FloatArray = npt.NDArray[np.float64]

MIN_HOURS_PER_DAY = 20
HORIZON_DAYS = 3
MIN_SEASON_WINDOWS = 40
MAX_OUTDOOR_C = 12.0  # days warmer than this are not heating days: flow means nothing
MIN_SEASONS = 2
MIN_GAIN = 0.05  # C of room per C of flow
MAX_GAIN = 0.8
MIN_RATE_PER_DAY = 1 / 30
MAX_RATE_PER_DAY = 1.0
MAX_GAIN_SPREAD = 4.0  # largest / smallest season gain
MAX_FLOW_COEFF_SPREAD = (
    5.0  # the same for gain * rate, which sets the first days' response
)


@dataclass(frozen=True)
class DailyData:
    """Daily means, one entry per day; NaN where a day has too little data."""

    days: list[datetime]
    room: FloatArray
    t_out: FloatArray
    flow: FloatArray
    sun: FloatArray


@dataclass(frozen=True)
class SeasonFit:
    season: int  # heating season starting in July of this year
    windows: int
    gain: float  # C of room per C of flow, in the long run
    rate_per_day: float  # share of the gap to equilibrium closed per day


@dataclass(frozen=True)
class DailyRoomFit:
    gain_c_per_c: float
    rate_per_day: float
    time_constant_days: float
    seasons: tuple[SeasonFit, ...]
    windows: int
    cv_rmse_c: dict[int, float]  # leave-one-season-out error 1-3 days ahead
    cv_persistence_rmse_c: dict[int, float]  # the same for "no change"


def daily_means(
    hours: list[datetime], values: FloatArray, min_hours: int = MIN_HOURS_PER_DAY
) -> FloatArray:
    """Mean per UTC date (in date order); NaN when fewer than `min_hours` hours are known."""
    by_day: dict[object, list[float]] = defaultdict(list)
    for moment, value in zip(hours, values, strict=True):
        if np.isfinite(value):
            by_day[moment.date()].append(float(value))
    dates = sorted({moment.date() for moment in hours})
    return np.array(
        [
            float(np.mean(by_day[d])) if len(by_day.get(d, [])) >= min_hours else np.nan
            for d in dates
        ]
    )


def _season(day: datetime) -> int:
    return day.year if day.month >= 7 else day.year - 1


class _Windows:
    """All starting days with complete data for the following HORIZON_DAYS days."""

    def __init__(self, data: DailyData):
        finite = (
            np.isfinite(data.room)
            & np.isfinite(data.t_out)
            & np.isfinite(data.flow)
            & np.isfinite(data.sun)
            & (np.nan_to_num(data.t_out, nan=99.0) <= MAX_OUTDOOR_C)
        )
        starts = np.array(
            [
                k
                for k in range(len(data.room) - HORIZON_DAYS)
                if finite[k : k + HORIZON_DAYS + 1].all()
            ],
            dtype=int,
        )
        steps = starts[:, None] + np.arange(HORIZON_DAYS)[None, :]
        self.season = np.array([_season(data.days[k]) for k in starts])
        self.room0 = data.room[starts]
        self.t_out = data.t_out[steps]
        self.flow = data.flow[steps]
        self.sun = data.sun[steps]
        self.target = {h: data.room[starts + h] for h in range(1, HORIZON_DAYS + 1)}

    def simulate(
        self, p: FloatArray, sel: npt.NDArray[np.bool_]
    ) -> dict[int, FloatArray]:
        k_out, k_sun, b, k_flow = p
        room = self.room0[sel].copy()
        out: dict[int, FloatArray] = {}
        for k in range(HORIZON_DAYS):
            room = (
                room
                + k_out * (self.t_out[sel, k] - room)
                + k_flow * (self.flow[sel, k] - room)
                + k_sun * self.sun[sel, k] / 100.0
                + b
            )
            out[k + 1] = room.copy()
        return out

    def fit(self, sel: npt.NDArray[np.bool_]) -> FloatArray:
        def residuals(p: FloatArray) -> FloatArray:
            sim = self.simulate(p, sel)
            return np.concatenate([sim[h] - self.target[h][sel] for h in sim])

        result = least_squares(
            residuals,
            [0.05, 0.05, 0.0, 0.05],
            bounds=([0.0, 0.0, -3.0, 0.0], [1.0, 2.0, 3.0, 1.0]),
        )
        return np.asarray(result.x, dtype=float)


def _gain_and_rate(p: FloatArray) -> tuple[float, float]:
    rate = float(p[0] + p[3])
    return (float(p[3] / rate) if rate > 0 else 0.0), rate


def fit_daily_room(data: DailyData) -> DailyRoomFit:
    """Fit season by season, check that the seasons agree, and cross-validate."""
    windows = _Windows(data)
    usable = [
        s
        for s in sorted(set(windows.season.tolist()))
        if int((windows.season == s).sum()) >= MIN_SEASON_WINDOWS
    ]
    if len(usable) < MIN_SEASONS:
        raise ModelFitError(
            f"need at least {MIN_SEASONS} seasons with {MIN_SEASON_WINDOWS} complete windows "
            f"to judge the room response, found {len(usable)}"
        )
    seasons = []
    for s in usable:
        sel = windows.season == s
        gain, rate = _gain_and_rate(windows.fit(sel))
        seasons.append(SeasonFit(s, int(sel.sum()), gain, rate))
    for fit in seasons:
        if fit.gain < MIN_GAIN:
            raise ModelFitError(
                f"no measurable response of the room to the flow temperature in season "
                f"{fit.season} (gain {fit.gain:.2f})"
            )
        if (
            fit.gain > MAX_GAIN
            or not MIN_RATE_PER_DAY <= fit.rate_per_day <= MAX_RATE_PER_DAY
        ):
            raise ModelFitError(
                f"implausible room response in season {fit.season} "
                f"(gain {fit.gain:.2f}, time constant {1 / fit.rate_per_day:.1f} days)"
            )
    gains = [f.gain for f in seasons]
    flow_coefficients = [f.gain * f.rate_per_day for f in seasons]
    if max(gains) > MAX_GAIN_SPREAD * min(gains) or max(
        flow_coefficients
    ) > MAX_FLOW_COEFF_SPREAD * min(flow_coefficients):
        raise ModelFitError(
            f"seasons differ too much in the room response to flow: "
            f"{', '.join(f'{f.season}: gain {f.gain:.2f}, rate {f.rate_per_day:.2f}/day' for f in seasons)}"
        )
    gain = float(np.median(gains))
    rate = float(np.median([f.rate_per_day for f in seasons]))

    errors: dict[int, list[float]] = {h: [] for h in range(1, HORIZON_DAYS + 1)}
    persistence: dict[int, list[float]] = {h: [] for h in range(1, HORIZON_DAYS + 1)}
    for s in usable:
        test = windows.season == s
        train = np.isin(windows.season, usable) & ~test
        if int(train.sum()) < MIN_SEASON_WINDOWS:
            continue
        sim = windows.simulate(windows.fit(train), test)
        for h in sim:
            errors[h] += (sim[h] - windows.target[h][test]).tolist()
            persistence[h] += (windows.room0[test] - windows.target[h][test]).tolist()
    return DailyRoomFit(
        gain_c_per_c=gain,
        rate_per_day=rate,
        time_constant_days=1.0 / rate,
        seasons=tuple(seasons),
        windows=int(sum(f.windows for f in seasons)),
        cv_rmse_c={
            h: float(np.sqrt(np.mean(np.square(v)))) for h, v in errors.items() if v
        },
        cv_persistence_rmse_c={
            h: float(np.sqrt(np.mean(np.square(v))))
            for h, v in persistence.items()
            if v
        },
    )


def room_params_from_fit(fit: DailyRoomFit) -> RoomParams:
    """Hourly planner parameters: floor heat couples directly to the room (no slab state),
    sun and the unmeasured heat gain are left out (the plan anchors the drift to the room's
    current temperature)."""
    per_hour = fit.rate_per_day / 24.0
    return RoomParams(
        tau_slab_h=1.0,
        k_slab=fit.gain_c_per_c * per_hour,
        k_out=(1.0 - fit.gain_c_per_c) * per_hour,
        k_sun=0.0,
        sun_lag_h=1.0,
        drift_c_per_h=0.0,
    )
