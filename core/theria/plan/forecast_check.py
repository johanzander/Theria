"""Plan against reality: every planned hour, once it is over, next to what was measured.

Each finished hour is compared with every stored plan run that planned it, grouped by how far ahead
the run was made (the lead: from when the plan was computed to the end of the hour). One JSON line per
hour is kept for good, so that the model can be improved where the errors are systematic. Measured:
the room at the end of the hour, the heat pump's energy, the room setpoint, the pump's own flow target,
the outdoor temperature and the sun (Open-Meteo's record of the hour). The plan side is what each run
expected or used for that hour. An hour without a measurement is kept with None and counted, so a
stopped InfluxDB shows instead of quietly leaving the check empty.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

import numpy as np

from core.theria.plan.calibration_data import (
    MAX_AUX_KWH_PER_HOUR,
    MAX_ENERGY_KWH_PER_HOUR,
    SETPOINT_FFILL_HOURS,
    SMALL_GAP_HOURS,
    HistorySource,
)
from core.theria.plan.series import (
    FloatArray,
    despike,
    hour_floor,
    hourly_counter_energy,
    hourly_grid,
    interpolate_gaps,
    to_hourly,
)
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import WeatherHour

logger = logging.getLogger(__name__)

QUANTITIES = {
    "room_c": "°C",
    "energy_kwh": "kWh",
    "rts_c": "°C",
    "flow_c": "°C",
    "t_out_c": "°C",
    "sun_wm2": "W/m²",
}
# (from, to] hours between the plan's computation and the end of the hour
LEADS = ((0, 1), (1, 3), (3, 6), (6, 12), (12, 24), (24, 48))
SETTLE = timedelta(hours=1)  # an hour is checked once it ended this long ago
REACH = timedelta(
    days=6
)  # no further back than the stored plan runs (7 days) can answer
ROOM_FFILL_HOURS = 3  # room sensors report on change
FLOW_FFILL_HOURS = 24  # the pump's flow target is logged only when it changes


def lead_label(lead: tuple[int, int]) -> str:
    return f"{lead[0]}-{lead[1]} h"


def _lead_of(hours: float) -> str | None:
    return next(
        (lead_label(lead) for lead in LEADS if lead[0] < hours <= lead[1]), None
    )


@dataclass(frozen=True)
class Measured:
    """Hourly measurements; NaN = not measured. room_c is the room at the end of the hour."""

    hours: list[datetime]
    room_c: FloatArray
    energy_kwh: FloatArray
    rts_c: FloatArray
    flow_c: FloatArray
    t_out_c: FloatArray
    sun_wm2: FloatArray

    def at(self, hour: datetime) -> dict[str, float | None] | None:
        if hour not in self.hours:
            return None
        i = self.hours.index(hour)
        values = {q: float(getattr(self, q)[i]) for q in QUANTITIES}
        return {q: (v if np.isfinite(v) else None) for q, v in values.items()}


def load_measured(
    source: HistorySource,
    settings: PlanSettings,
    weather_history: Callable[[datetime, datetime], list[WeatherHour]],
    start: datetime,
    stop: datetime,
) -> Measured:
    first = hour_floor(start)
    n = int((hour_floor(stop) - first).total_seconds() // 3600)
    hours = hourly_grid(first, n)

    def series(entity: str, field: str | None = None) -> Any:
        return source.series(entity, first, stop, field)

    def counter(entity: str, max_kwh_per_hour: float) -> FloatArray:
        return hourly_counter_energy(
            series(entity), first, n, max_kwh_per_hour, SMALL_GAP_HOURS
        )

    sun = {w.start: w.shortwave_wm2 for w in weather_history(first, stop)}
    return Measured(
        hours=hours,
        room_c=np.nanmean(
            [
                to_hourly(series(e), first, n, how="last", ffill_hours=ROOM_FFILL_HOURS)
                for e in settings.room_temp_entities
            ],
            axis=0,
        ),
        energy_kwh=counter(settings.energy_counter_entity, MAX_ENERGY_KWH_PER_HOUR)
        + counter(settings.aux_energy_entity, MAX_AUX_KWH_PER_HOUR),
        rts_c=to_hourly(
            series(settings.room_setpoint_entity, "temperature"),
            first,
            n,
            how="last",
            ffill_hours=SETPOINT_FFILL_HOURS,
        ),
        flow_c=to_hourly(
            series(settings.flow_setpoint_entity),
            first,
            n,
            ffill_hours=FLOW_FFILL_HOURS,
        ),
        t_out_c=interpolate_gaps(
            despike(to_hourly(series(settings.outdoor_entity), first, n), 7, 6.0), 3
        ),
        sun_wm2=np.array([sun.get(h, np.nan) for h in hours], dtype=float),
    )


def _planned(run: dict[str, Any], hour: datetime) -> dict[str, float] | None:
    """What a stored run expected or used for `hour`; None when it did not plan it."""
    starts = run["inputs"]["hour_starts"]
    key = hour.isoformat()
    if key not in starts:
        return None
    i = starts.index(key)
    decided = run["decided"][i]
    values = {
        "room_c": decided.get("room_c"),  # older runs did not keep it
        "energy_kwh": decided["energy_kwh"],
        "rts_c": decided["rts_c"],
        "flow_c": decided.get("flow_c"),
        "t_out_c": run["inputs"]["t_out_c"][i],
        "sun_wm2": run["inputs"]["sun_wm2"][i],
    }
    return {q: float(v) for q, v in values.items() if v is not None}


def check_hour(
    hour: datetime,
    runs: Sequence[dict[str, Any]],
    measured: dict[str, float | None],
) -> dict[str, Any]:
    by_lead: dict[str, list[dict[str, float]]] = {}
    end = hour + timedelta(hours=1)
    for run in runs:
        planned = _planned(run, hour)
        if planned is None:
            continue
        computed_at = datetime.fromisoformat(run["computed_at"])
        lead = _lead_of((end - computed_at) / timedelta(hours=1))
        if lead is not None:
            by_lead.setdefault(lead, []).append(planned)
    predicted = {
        lead: {
            q: float(np.mean(values))
            for q in QUANTITIES
            if (values := [p[q] for p in plans if q in p])
        }
        for lead, plans in by_lead.items()
    }
    return {
        "hour": hour.isoformat(),
        "measured": measured,
        "predicted": predicted,
        "runs": {lead: len(plans) for lead, plans in by_lead.items()},
    }


def summarize(lines: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per quantity and lead: how many hours, the mean error (planned minus measured) and the mean
    absolute error; and per quantity how many hours had no measurement."""
    errors: dict[tuple[str, str], list[float]] = {}
    missing = dict.fromkeys(QUANTITIES, 0)
    for line in lines:
        for q in QUANTITIES:
            actual = line["measured"][q]
            if actual is None:
                missing[q] += 1
                continue
            for lead, planned in line["predicted"].items():
                if q in planned:
                    errors.setdefault((q, lead), []).append(planned[q] - actual)
    order = [lead_label(lead) for lead in LEADS]
    rows = [
        {
            "quantity": q,
            "unit": QUANTITIES[q],
            "lead": lead,
            "n": len(values),
            "bias": float(np.mean(values)),
            "mae": float(np.mean(np.abs(values))),
        }
        for q in QUANTITIES
        for lead in order
        if (values := errors.get((q, lead)))
    ]
    return {"hours": len(lines), "rows": rows, "missing": missing}


class ForecastCheckStore:
    """One JSON line per checked hour, kept for good."""

    def __init__(self, path: str):
        self._path = path

    def _all(self) -> list[dict[str, Any]]:
        try:
            with open(self._path) as f:
                return [json.loads(line) for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def read(self, since: datetime) -> list[dict[str, Any]]:
        return [
            line
            for line in self._all()
            if datetime.fromisoformat(line["hour"]) >= since
        ]

    def last_hour(self) -> datetime | None:
        lines = self._all()
        return datetime.fromisoformat(lines[-1]["hour"]) if lines else None

    def append(self, lines: Sequence[dict[str, Any]]) -> None:
        with open(self._path, "a") as f:
            f.writelines(json.dumps(line) + "\n" for line in lines)


class RunReader(Protocol):
    def read(self, since: datetime) -> list[dict[str, Any]]: ...


class ForecastChecker:
    def __init__(
        self,
        runs: RunReader,
        measured: Callable[[datetime, datetime], Measured],
        store: ForecastCheckStore,
        now: Callable[[], datetime],
    ):
        self._runs = runs
        self._measured = measured
        self._store = store
        self._now = now

    def run_due(self) -> int:
        """Check every hour that has finished since the last check; returns how many."""
        now = self._now()
        last_due = hour_floor(now - SETTLE) - timedelta(hours=1)
        last = self._store.last_hour()
        first = hour_floor(now - REACH)
        if last is not None:
            first = max(first, last + timedelta(hours=1))
        if first > last_due:
            return 0
        stop = last_due + timedelta(hours=1)
        measured = self._measured(first, stop)
        runs = self._runs.read(first - timedelta(hours=LEADS[-1][1]))
        lines = []
        hour = first
        while hour < stop:
            values = measured.at(hour) or dict.fromkeys(QUANTITIES)
            lines.append(check_hour(hour, runs, values))
            hour += timedelta(hours=1)
        self._store.append(lines)
        gaps = [q for q in ("room_c", "energy_kwh") if lines[-1]["measured"][q] is None]
        if gaps:
            logger.warning(
                "no measured %s for %s: the plan cannot be checked against it",
                " or ".join(gaps),
                lines[-1]["hour"],
            )
        return len(lines)
