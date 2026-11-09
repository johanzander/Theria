"""The heat pump's electricity for the next day: heating from the weather and the setpoint, hot water from the clock.

Chosen and reviewed in `documentation/energy-forecast/README.md`, on a walk-forward test over four years. Each day is
forecast from what was known at the moment of the forecast. The compressor only: the electric heater's counter cannot
be read. The forecasts are kept as they were made, and two simple references run next to the model.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.calibration_data import (
    FLOW_PLAUSIBLE_C,
    HistorySource,
)
from core.theria.plan.series import (
    FloatArray,
    despike,
    hourly_grid,
    interpolate_gaps,
    interpolated_counter_energy,
    to_hourly,
)
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import WeatherHour

HALF_LIFE_DAYS = 365.0  # older hours count for less
RIDGE = 1.0
TRAINING_DAYS = 1095
MIN_TRAINING_HOURS = 14 * 24
FLOW_FIT_HOURS = 60 * 24  # the pump's curve as it stands now
MIN_FLOW_ROWS = 200
LEVEL_DAYS = 3  # how far back the level of the last days is read
LEVEL_PRIOR_KWH = 5.0
LEVEL_CLIP = (0.5, 2.0)
LEVEL_POWER = 0.5  # half of what the last days say; the rest is noise
HOT_WATER_DAYS = 28
PROFILE_DAYS = 7
MAX_HOURLY_KWH = 10.0  # above this an hour is a counter error
MAX_COUNTER_KWH_PER_HOUR = 8.0
MAX_COUNTER_GAP_HOURS = 8
ISSUE_HOUR = 14  # local: tomorrow's prices are out
SETPOINT_FFILL_HOURS = 400 * 24  # logged only when it changes
FLOW_FFILL_HOURS = 6
OUTDOOR_GAP_HOURS = 3
OUTDOOR_PLAUSIBLE_C = (-35.0, 45.0)  # the sensor sometimes reports garbage
BACKFILL_DAYS = 14
# `Min. utetemperatur` stood at -20: the flow target was about 2.6 C high in this period
CURVE_CHANGED = (
    datetime(2026, 11, 8, 11, 41, tzinfo=UTC),
    datetime(2026, 11, 9, 16, 25, tzinfo=UTC),
)


@dataclass
class EnergyHistory:
    """Aligned hourly arrays; NaN = unknown. Weather and setpoint may run past the measurements."""

    hours: list[datetime]
    t_out_c: FloatArray
    sun_wm2: FloatArray
    wind_kmh: FloatArray
    setpoint_c: FloatArray
    flow_c: FloatArray
    heating_kwh: FloatArray
    hot_water_kwh: FloatArray


def _trailing(values: FloatArray, hours: int) -> FloatArray:
    """The mean of the known values in the last `hours`; NaN where none is known."""
    known = np.isfinite(values)
    summed = np.concatenate([[0.0], np.cumsum(np.where(known, values, 0.0))])
    counted = np.concatenate([[0], np.cumsum(known)])
    first = np.maximum(np.arange(1, len(values) + 1) - hours, 0)
    last = np.arange(1, len(values) + 1)
    count = counted[last] - counted[first]
    out = np.full(len(values), np.nan)
    seen = count > 0
    out[seen] = (summed[last] - summed[first])[seen] / count[seen]
    return out


def _hinge(knot: float, values: FloatArray) -> FloatArray:
    return np.maximum(knot - values, 0.0)


class _Heating:
    """The heating model's inputs for one history."""

    def __init__(self, h: EnergyHistory, cut: int):
        self.h = h
        self.t = h.t_out_c
        self.t24 = _trailing(h.t_out_c, 24)
        self.t72 = _trailing(h.t_out_c, 72)
        self.sun = np.nan_to_num(h.sun_wm2)
        self.wind = np.nan_to_num(h.wind_kmh)
        changed = np.array([CURVE_CHANGED[0] <= x < CURVE_CHANGED[1] for x in h.hours])
        self.changed = changed
        self.flow_fit = self._fit_flow(cut)

    def _flow_x(self, i: np.ndarray) -> np.ndarray:
        t = self.t[i]
        return np.column_stack([np.ones(len(i)), t, t * t, self.h.setpoint_c[i] - 18.0])

    def _fit_flow(self, cut: int) -> np.ndarray:
        known = np.isfinite(self.h.flow_c) & ~self.changed & np.isfinite(self.t)
        known &= np.isfinite(self.h.setpoint_c)
        known[cut:] = False
        recent = known.copy()
        recent[: max(0, cut - FLOW_FIT_HOURS)] = False
        rows = np.flatnonzero(recent if recent.sum() >= MIN_FLOW_ROWS else known)
        if len(rows) == 0:
            raise ModelFitError("no flow target history to fit the pump's curve on")
        coefficients, *_ = np.linalg.lstsq(
            self._flow_x(rows), self.h.flow_c[rows], rcond=None
        )
        return np.asarray(coefficients)

    def estimated_flow(self, i: np.ndarray) -> np.ndarray:
        return np.asarray(self._flow_x(i) @ self.flow_fit)

    def features(self, i: np.ndarray, flow: np.ndarray) -> np.ndarray:
        t = self.t[i]
        degrees = _hinge(16.0, t)
        over = flow - 35.0
        sun = self.sun[i] / 100.0
        return np.column_stack(
            [
                np.ones(len(i)),
                degrees,
                _hinge(10.0, t),
                _hinge(4.0, t),
                _hinge(-2.0, t),
                _hinge(16.0, self.t24[i]),
                _hinge(16.0, self.t72[i]),
                over,
                over * degrees / 10.0,
                sun,
                sun * degrees / 10.0,
                self.wind[i] / 10.0 * degrees / 10.0,
            ]
        )


def _ridge(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    a = x.T @ (x * weights[:, None])
    penalty = RIDGE * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.asarray(np.linalg.solve(a + penalty, x.T @ (y * weights)))


def _trimmed_mean(values: np.ndarray) -> float:
    ordered = np.sort(values)
    if len(ordered) >= 5:
        ordered = ordered[1:-1]
    elif len(ordered) >= 3:
        ordered = ordered[:-1]
    return float(ordered.mean()) if len(ordered) else 0.0


def forecast_energy(
    h: EnergyHistory, cut: int, target: np.ndarray, timezone: str
) -> dict[str, FloatArray]:
    """kWh per hour of `target`, per method, from what was measured before index `cut`."""
    tz = ZoneInfo(timezone)
    local = [x.astimezone(tz) for x in h.hours]
    clock = np.array([x.hour for x in local])
    ordinal = np.array([x.date().toordinal() for x in local])
    weekend = np.array([x.weekday() >= 5 for x in local])
    day = int(ordinal[target[0]])
    before = np.arange(len(h.hours)) < cut
    total = h.heating_kwh + h.hot_water_kwh
    valid = (
        np.isfinite(total)
        & (h.heating_kwh >= 0)
        & (h.hot_water_kwh >= 0)
        & (total <= MAX_HOURLY_KWH)
        & before
    )
    heating = _Heating(h, cut)
    usable = valid & np.isfinite(h.t_out_c) & np.isfinite(h.setpoint_c)
    usable &= ~heating.changed & np.isfinite(heating.t24) & np.isfinite(heating.t72)
    rows = np.flatnonzero(usable)
    if len(rows) < MIN_TRAINING_HOURS:
        raise ModelFitError(
            f"energy forecast needs {MIN_TRAINING_HOURS} hours of history, found {len(rows)}"
        )
    flow = np.where(
        np.isfinite(h.flow_c[rows]), h.flow_c[rows], heating.estimated_flow(rows)
    )
    weights = 0.5 ** ((cut - rows) / (24.0 * HALF_LIFE_DAYS))
    beta = _ridge(heating.features(rows, flow), h.heating_kwh[rows], weights)

    def model(i: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.maximum(heating.features(i, heating.estimated_flow(i)) @ beta, 0.0)
        )

    heat = model(target)
    recent = np.flatnonzero(usable & (ordinal >= day - LEVEL_DAYS - 1))
    if len(recent) > LEVEL_DAYS * 12:
        ratio = (h.heating_kwh[recent].sum() + LEVEL_PRIOR_KWH) / (
            model(recent).sum() + LEVEL_PRIOR_KWH
        )
        heat = heat * float(np.clip(ratio, *LEVEL_CLIP)) ** LEVEL_POWER

    same_kind = valid & (ordinal >= day - HOT_WATER_DAYS - 1)
    same_kind &= weekend == weekend[target[0]]
    hot_by_hour = np.array(
        [
            h.hot_water_kwh[same_kind & (clock == hour)].mean()
            if (same_kind & (clock == hour)).any()
            else 0.0
            for hour in range(24)
        ]
    )
    week = valid & (ordinal >= day - PROFILE_DAYS - 1)
    profile = np.array(
        [_trimmed_mean(total[week & (clock == hour)]) for hour in range(24)]
    )
    last_48 = np.flatnonzero(valid)[-48:]
    return {
        "simple": heat + hot_by_hour[clock[target]],
        "mean_48h": np.full(len(target), float(total[last_48].mean())),
        "profile_7d": profile[clock[target]],
    }


def load_energy_history(
    source: HistorySource,
    settings: PlanSettings,
    weather_history: Callable[[datetime, datetime], list[WeatherHour]],
    start: datetime,
    stop: datetime,
) -> EnergyHistory:
    n = int((stop - start) / timedelta(hours=1))
    hours = hourly_grid(start, n)
    now = min(stop, datetime.now(UTC))
    weather = {w.start: w for w in weather_history(start, now)}

    def counter(entity: str) -> FloatArray:
        return interpolated_counter_energy(
            source.series(entity, start, now),
            start,
            n,
            MAX_COUNTER_KWH_PER_HOUR,
            MAX_COUNTER_GAP_HOURS,
        )

    def from_weather(field: str) -> FloatArray:
        return np.array(
            [getattr(weather[x], field) if x in weather else np.nan for x in hours],
            dtype=float,
        )

    outdoor = interpolate_gaps(
        despike(
            to_hourly(
                source.hourly_means(
                    settings.outdoor_entity, start, now, OUTDOOR_PLAUSIBLE_C
                ),
                start,
                n,
            ),
            7,
            6.0,
        ),
        OUTDOOR_GAP_HOURS,
    )
    archived = from_weather("temperature_c")
    both = np.isfinite(outdoor) & np.isfinite(archived)
    shift = float(np.mean(outdoor[both] - archived[both])) if both.any() else 0.0
    flow = to_hourly(
        [
            p
            for p in source.series(settings.flow_setpoint_entity, start, now)
            if FLOW_PLAUSIBLE_C[0] <= p[1] <= FLOW_PLAUSIBLE_C[1]
        ],
        start,
        n,
        how="last",
        ffill_hours=FLOW_FFILL_HOURS,
    )
    return EnergyHistory(
        hours=hours,
        t_out_c=np.where(np.isfinite(outdoor), outdoor, archived + shift),
        sun_wm2=from_weather("shortwave_wm2"),
        wind_kmh=from_weather("wind_kmh"),
        setpoint_c=to_hourly(
            source.series(settings.room_setpoint_entity, start, now, "temperature"),
            start,
            n,
            how="last",
            ffill_hours=SETPOINT_FFILL_HOURS,
        ),
        flow_c=flow,
        heating_kwh=counter(settings.energy_counter_entity),
        hot_water_kwh=counter(settings.hot_water_energy_entity),
    )


def summarize_days(days: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per method and kind of weather: the mean error of the day's total, its bias and its share of the use."""
    errors: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for line in days:
        actual = line["actual"]
        if any(value is None for value in actual):
            continue
        measured = float(sum(actual))
        for method, forecast in line["forecasts"].items():
            errors.setdefault((method, line["weather"]), []).append(
                (float(sum(forecast)) - measured, measured)
            )
    rows = []
    for (method, weather), pairs in sorted(errors.items()):
        difference = np.array([e for e, _ in pairs])
        rows.append(
            {
                "method": method,
                "weather": weather,
                "days": len(pairs),
                "day_error_kwh": float(np.abs(difference).mean()),
                "bias_kwh": float(difference.mean()),
                "share": float(
                    np.abs(difference).sum() / max(sum(m for _, m in pairs), 1e-9)
                ),
            }
        )
    return rows


class EnergyForecastStore:
    """One JSON line per forecast day, as it was made, kept for good."""

    def __init__(self, path: str):
        self._path = path

    def read(self) -> list[dict[str, Any]]:
        try:
            with open(self._path) as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def append(self, lines: Sequence[dict[str, Any]]) -> None:
        with open(self._path, "a") as f:
            f.writelines(json.dumps(line) + "\n" for line in lines)


class EnergyForecaster:
    """Issues tomorrow's forecast once a day, and fills in the last days from what was measured."""

    def __init__(
        self,
        load: Callable[[datetime, datetime], EnergyHistory],
        weather: Callable[[], list[WeatherHour]],
        store: EnergyForecastStore,
        timezone: str,
        now: Callable[[], datetime],
        planned_setpoint: Callable[[datetime], float | None],
        backfill_days: int = BACKFILL_DAYS,
    ):
        self._load = load
        self._weather = weather
        self._store = store
        self._timezone = timezone
        self._tz = ZoneInfo(timezone)
        self._now = now
        self._planned_setpoint = planned_setpoint
        self._backfill_days = backfill_days

    def _day_start(self, day: date) -> datetime:
        return datetime(day.year, day.month, day.day, tzinfo=self._tz).astimezone(UTC)

    def run_due(self) -> int:
        """Make the forecasts that are due; returns how many."""
        now = self._now().astimezone(self._tz)
        today = now.date()
        have = {line["day"] for line in self._store.read()}
        past = [today - timedelta(days=k) for k in range(self._backfill_days, 0, -1)]
        missing = [d for d in past if d.isoformat() not in have]
        tomorrow = today + timedelta(days=1)
        issue = now.hour >= ISSUE_HOUR and tomorrow.isoformat() not in have
        if not missing and not issue:
            return 0
        first = (missing[0] if missing else tomorrow) - timedelta(days=TRAINING_DAYS)
        h = self._load(
            self._day_start(first), self._day_start(tomorrow + timedelta(days=1))
        )
        lines = [self._measured_day(h, day) for day in missing]
        if issue:
            lines.append(self._tomorrow(h, tomorrow))
        self._store.append(lines)
        return len(lines)

    def _indices(self, h: EnergyHistory, day: date) -> np.ndarray:
        return np.array(
            [i for i, x in enumerate(h.hours) if x.astimezone(self._tz).date() == day]
        )

    def _line(
        self, h: EnergyHistory, day: date, target: np.ndarray, weather: str, cut: int
    ) -> dict[str, Any]:
        forecasts = forecast_energy(h, cut, target, self._timezone)
        return {
            "day": day.isoformat(),
            "issued_at": self._now().isoformat(),
            "weather": weather,
            "hours": [h.hours[i].isoformat() for i in target],
            "clock": [h.hours[i].astimezone(self._tz).hour for i in target],
            "forecasts": {k: [float(x) for x in v] for k, v in forecasts.items()},
        }

    def _measured_day(self, h: EnergyHistory, day: date) -> dict[str, Any]:
        """A day already over, as if forecast at 14:00 the day before, with the weather that came."""
        target = self._indices(h, day)
        before = day - timedelta(days=1)
        cut = next(
            i
            for i, x in enumerate(h.hours)
            if x.astimezone(self._tz).date() == before
            and x.astimezone(self._tz).hour == ISSUE_HOUR
        )
        return self._line(h, day, target, "measured", cut)

    def _tomorrow(self, h: EnergyHistory, day: date) -> dict[str, Any]:
        moment = self._now().astimezone(UTC).replace(minute=0, second=0, microsecond=0)
        cut = h.hours.index(moment)
        ahead = {w.start: w for w in self._weather()}
        past = [i for i in range(cut - 24, cut) if h.hours[i] in ahead]
        errors = [h.t_out_c[i] - ahead[h.hours[i]].temperature_c for i in past]
        shift = float(np.nanmean(errors)) if errors else 0.0
        held = float(h.setpoint_c[cut - 1])
        for i in range(cut, len(h.hours)):
            hour = h.hours[i]
            if hour in ahead:
                h.t_out_c[i] = ahead[hour].temperature_c + shift
                h.sun_wm2[i] = ahead[hour].shortwave_wm2
                h.wind_kmh[i] = ahead[hour].wind_kmh
            planned = self._planned_setpoint(hour)
            h.setpoint_c[i] = held if planned is None else planned
        target = self._indices(h, day)
        if not np.isfinite(h.t_out_c[cut : target[-1] + 1]).all():
            raise ModelFitError(f"no weather forecast up to the end of {day}")
        return self._line(h, day, target, "forecast", cut)

    def view(self, days: int) -> dict[str, Any]:
        """The forecasts of the last `days` next to what was measured, and the error."""
        tomorrow = self._now().astimezone(self._tz).date() + timedelta(days=1)
        first = tomorrow - timedelta(days=days)
        lines = [
            line for line in self._store.read() if line["day"] >= first.isoformat()
        ]
        lines.sort(key=lambda line: line["day"])
        if not lines:
            return {"days": [], "summary": []}
        h = self._load(
            self._day_start(first - timedelta(days=1)),
            self._day_start(tomorrow + timedelta(days=1)),
        )
        total = h.heating_kwh + h.hot_water_kwh
        by_hour = dict(zip(h.hours, total, strict=True))
        now = self._now()
        out = []
        for line in lines:
            values = [
                by_hour.get(datetime.fromisoformat(x), float("nan"))
                for x in line["hours"]
            ]
            actual = [
                float(v)
                if np.isfinite(v)
                and datetime.fromisoformat(x) + timedelta(hours=1) <= now
                else None
                for x, v in zip(line["hours"], values, strict=True)
            ]
            out.append(
                {
                    **{k: line[k] for k in ("day", "weather", "clock", "forecasts")},
                    "actual": actual,
                }
            )
        return {"days": out, "summary": summarize_days(out)}
