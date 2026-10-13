"""What the plan achieved: heating cost against a baseline, per day, from real meters.

The pump's electricity counters (compressor and electric heater) and the hourly price give
what the heating actually cost. The baseline cost is what the same hours would have cost at
the base setpoint: each hour's measured energy scaled by the pump model's relative change
for the setpoint (so weather and house are real, only the setpoint effect is modelled). The
electric heater's energy is treated conservatively: in an hour where the setpoint was raised
above the base all of it is our cost (the baseline would not have started it), in other hours
it is the same in the baseline, so a boost that makes the heater start shows up as lost
saving. Hours with hot water are left out because the plan does not control them.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import numpy.typing as npt

from core.theria.plan.calibration_data import MAX_ENERGY_KWH_PER_HOUR, HistorySource
from core.theria.plan.pump_model import PumpParams, energy_kwh_per_hour
from core.theria.plan.series import (
    despike,
    hour_floor,
    hourly_energy_from_counter,
    hourly_grid,
    interpolate_gaps,
    step_fraction,
    to_hourly,
)
from core.theria.plan.settings import PlanSettings

FloatArray = npt.NDArray[np.float64]

SETPOINT_FFILL_HOURS = 24 * 60  # the setpoint is only logged when it changes
MIN_MODEL_KWH = 0.2  # keeps the scaling from exploding when the model energy is tiny
MIN_SCALE, MAX_SCALE = 0.5, 2.0
MOSTLY_HOT_WATER = (
    0.5  # hours with more hot water than this are not watched for heating trouble
)


@dataclass(frozen=True)
class OutcomeInputs:
    """Aligned hourly series; NaN = unknown."""

    hours: list[datetime]
    price_sek_per_kwh: FloatArray
    rts_c: FloatArray
    t_out_c: FloatArray
    compressor_kwh: FloatArray
    aux_kwh: FloatArray
    hot_water: FloatArray  # share of the hour with hot water (0 = all heating)
    room_c: FloatArray  # mean of the ground-floor comfort sensors
    # floor loops open per room (count, fractional within an hour); empty without actuator sensors
    loops_open: dict[str, FloatArray] = field(default_factory=dict)


@dataclass(frozen=True)
class DayOutcome:
    day: date
    heating_hours: int  # hours with complete data that count
    coverage: float  # heating_hours / hours of the day present
    compressor_kwh: float
    aux_kwh: float
    actual_cost_sek: float
    baseline_cost_sek: float
    saving_sek: float
    average_price_paid_sek_per_kwh: float
    average_price_baseline_sek_per_kwh: float
    mean_rts_c: float
    room_min_c: float  # NaN when the day has no room reading
    room_mean_c: float
    room_at_morning_c: float = float(
        "nan"
    )  # room when comfort starts (NaN: not measured)


def day_outcomes(
    inputs: OutcomeInputs,
    pump: PumpParams,
    rts_base_c: float | FloatArray,
    comfort_hour: Callable[[datetime], bool] = lambda moment: True,
    morning_hour: Callable[[datetime], bool] = lambda moment: False,
) -> list[DayOutcome]:
    """One outcome per UTC day in the inputs.

    `comfort_hour` says which hours count for the room minimum (comfort is not needed at night).
    """
    base = np.broadcast_to(np.asarray(rts_base_c, dtype=float), (len(inputs.hours),))
    model_actual = np.maximum(
        np.asarray(
            energy_kwh_per_hour(pump, inputs.t_out_c, inputs.rts_c), dtype=float
        ),
        MIN_MODEL_KWH,
    )
    model_base = np.maximum(
        np.asarray(energy_kwh_per_hour(pump, inputs.t_out_c, base), dtype=float),
        MIN_MODEL_KWH,
    )
    scale = np.clip(model_base / model_actual, MIN_SCALE, MAX_SCALE)
    counts = (
        np.isfinite(inputs.price_sek_per_kwh)
        & np.isfinite(inputs.rts_c)
        & np.isfinite(inputs.t_out_c)
        & np.isfinite(inputs.compressor_kwh)
        & np.isfinite(inputs.aux_kwh)
        & (inputs.hot_water == 0.0)
    )
    watched = np.isfinite(inputs.aux_kwh) & (inputs.hot_water < MOSTLY_HOT_WATER)
    by_day: dict[date, list[int]] = defaultdict(list)
    watched_by_day: dict[date, list[int]] = defaultdict(list)
    hours_in_day: dict[date, int] = defaultdict(int)
    # hours the pump spent on hot water are not missing data: the coverage is of the others
    heating_hours_in_day: dict[date, int] = defaultdict(int)
    for i, moment in enumerate(inputs.hours):
        hours_in_day[moment.date()] += 1
        if not inputs.hot_water[i] > 0.0:
            heating_hours_in_day[moment.date()] += 1
        if counts[i]:
            by_day[moment.date()].append(i)
        if watched[i]:
            watched_by_day[moment.date()].append(i)
    outcomes = []
    for day in sorted(hours_in_day):
        idx = np.array(by_day.get(day, []), dtype=int)
        price = inputs.price_sek_per_kwh[idx]
        compressor = inputs.compressor_kwh[idx]
        aux = inputs.aux_kwh[idx]
        actual_kwh = compressor + aux
        raised = inputs.rts_c[idx] > base[idx] + 1e-9
        baseline_kwh = compressor * scale[idx] + np.where(raised, 0.0, aux)
        actual_cost = float(np.sum(price * actual_kwh))
        baseline_cost = float(np.sum(price * baseline_kwh))
        seen = np.array(watched_by_day.get(day, []), dtype=int)
        day_rooms = np.array(
            [
                inputs.room_c[i]
                for i, m in enumerate(inputs.hours)
                if m.date() == day and comfort_hour(m)
            ]
        )
        morning = np.array(
            [
                inputs.room_c[i]
                for i, m in enumerate(inputs.hours)
                if m.date() == day and morning_hour(m)
            ]
        )
        morning = morning[np.isfinite(morning)]
        known = day_rooms[np.isfinite(day_rooms)]
        room_min = float(np.min(known)) if len(known) else float("nan")
        room_mean = float(np.mean(known)) if len(known) else float("nan")
        outcomes.append(
            DayOutcome(
                day=day,
                heating_hours=len(idx),
                coverage=(
                    len(idx) / heating_hours_in_day[day]
                    if heating_hours_in_day[day]
                    else 0.0
                ),
                compressor_kwh=float(np.sum(compressor)),
                aux_kwh=float(np.sum(inputs.aux_kwh[seen])),
                actual_cost_sek=actual_cost,
                baseline_cost_sek=baseline_cost,
                saving_sek=baseline_cost - actual_cost,
                average_price_paid_sek_per_kwh=_average(actual_cost, actual_kwh),
                average_price_baseline_sek_per_kwh=_average(
                    baseline_cost, baseline_kwh
                ),
                mean_rts_c=(
                    float(np.mean(inputs.rts_c[idx])) if len(idx) else float("nan")
                ),
                room_min_c=room_min,
                room_mean_c=room_mean,
                room_at_morning_c=float(morning[0]) if len(morning) else float("nan"),
            )
        )
    return outcomes


def _average(cost: float, kwh: FloatArray) -> float:
    total = float(np.sum(kwh))
    return cost / total if total > 0 else float("nan")


def load_outcome_inputs(
    source: HistorySource, settings: PlanSettings, start: datetime, stop: datetime
) -> OutcomeInputs:
    """Aligned hourly series for [start, stop) (hours in UTC, timezone-aware inputs)."""
    first = hour_floor(start)
    n = int((hour_floor(stop) - first).total_seconds() // 3600)

    def mean(entity: str) -> FloatArray:
        return to_hourly(source.series(entity, first, stop), first, n, how="mean")

    def counter(entity: str) -> FloatArray:
        last = to_hourly(
            source.series(entity, first, stop), first, n, how="last", ffill_hours=3
        )
        return hourly_energy_from_counter(last, MAX_ENERGY_KWH_PER_HOUR)

    return OutcomeInputs(
        hours=hourly_grid(first, n),
        price_sek_per_kwh=mean(settings.price_entity),
        rts_c=to_hourly(
            source.series(settings.room_setpoint_entity, first, stop, "temperature"),
            first,
            n,
            how="last",
            ffill_hours=SETPOINT_FFILL_HOURS,
        ),
        t_out_c=interpolate_gaps(despike(mean(settings.outdoor_entity), 7, 6.0), 3),
        compressor_kwh=counter(settings.energy_counter_entity),
        aux_kwh=counter(settings.aux_energy_entity),
        hot_water=step_fraction(
            source.series(settings.heating_mode_entity, first, stop), first, n
        ),
        room_c=np.nanmean([mean(e) for e in settings.room_temp_entities], axis=0),
        loops_open={
            room: np.sum(
                [
                    step_fraction(source.series(e, first, stop), first, n)
                    for e in entities
                ],
                axis=0,
            )
            for room, entities in settings.room_actuators.items()
        },
    )
