"""One whole local day for the charts: what was measured so far and what is planned for the rest.

Every hour of the day is listed, so the chart always spans the day from 00:00 to 24:00. Hours that
have passed carry the measured setpoint and room temperature, the hours from now on carry the
plan; prices come from the price feed for both, hourly and per quarter.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.plan_model import HourPlan, Plan
from core.theria.plan.plan_record import PlanRecord
from core.theria.plan.prices import QUARTERS_PER_HOUR, PricePeriod
from core.theria.plan.series import hour_floor
from core.theria.plan.settings import PlanSettings
from core.theria.plan.shift_rules import thermostat_target
from core.theria.plan.zone_heat import (
    EXPECTED_HEATING_SHARE,
    ZoneMeasured,
    delivered_ground_kw,
    expected_open_share,
    ground_need_kw,
    measured_open_share,
    salon_need_kw,
)


def day_hours(day: date, timezone: str) -> list[datetime]:
    """The UTC hour starts of the local calendar day."""
    zone = ZoneInfo(timezone)
    first = datetime.combine(day, time(), zone).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time(), zone).astimezone(UTC)
    hours = []
    while first < end:
        hours.append(first)
        first += timedelta(hours=1)
    return hours


def _quarter_prices(
    prices: dict[datetime, float], hour: datetime
) -> list[float | None]:
    """The four 15-minute prices of the hour; None for a quarter without a price."""
    return [
        prices.get(hour + timedelta(minutes=15 * i)) for i in range(QUARTERS_PER_HOUR)
    ]


def _hour_price(prices: dict[datetime, float], hour: datetime) -> float | None:
    quarters = [hour + timedelta(minutes=15 * i) for i in range(QUARTERS_PER_HOUR)]
    if not all(q in prices for q in quarters):
        return None
    return float(np.mean([prices[q] for q in quarters]))


def _known(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _saving(
    price: float | None, baseline_kwh: float | None, kwh: float | None
) -> float | None:
    """What the hour's energy was worth against the flat setpoint's, at the hour's price."""
    if price is None or baseline_kwh is None or kwh is None:
        return None
    return float(price * (baseline_kwh - kwh))


def _expected_zones(
    hour: HourPlan, settings: PlanSettings, night: bool, salon_now_c: float | None
) -> dict[str, Any]:
    """The salon is where it is now plus what the plan changes: its forecast does not beat that."""
    room = hour.room_c_end + hour.sun_swing_c
    salon = None if salon_now_c is None else salon_now_c + hour.salon_delta_c
    return {
        "thermostat": hour.thermostat,
        "thermostat_target_c": thermostat_target(
            hour.thermostat, night, room, settings
        ),
        "loops_expected": expected_open_share(hour.thermostat, room, night, settings),
        "need_ground_kw": ground_need_kw(room, hour.t_out_c),
        "delivered_ground_kw": delivered_ground_kw(
            hour.flow_target_c,
            room,
            expected_open_share(hour.thermostat, room, night, settings),
            EXPECTED_HEATING_SHARE,
            settings.floor_mix_max_c,
        ),
        "salon_planned": salon,
        "need_salon_kw": None if salon is None else salon_need_kw(salon, hour.t_out_c),
    }


def build_day_view(
    day: date,
    now: datetime,
    settings: PlanSettings,
    periods: list[PricePeriod],
    plan: Plan | None,
    measured: OutcomeInputs | None,
    records: dict[datetime, PlanRecord] | None = None,
    zones: ZoneMeasured | None = None,
    salon_now_c: float | None = None,
) -> dict[str, Any]:
    """`measured` covers the hours before now; None when it could not be read.

    Every hour carries the plan, the flat baseline and what was measured side by side: for hours that
    have passed the plan is the one that was applied (from `records`), for the rest the latest plan.
    """
    current = hour_floor(now)
    records = records or {}
    prices = {p.start: p.price_sek_per_kwh for p in periods}
    planned = (
        {}
        if plan is None
        else {
            h.start: (h, b)
            for h, b in zip(plan.hours, plan.baseline_hours, strict=True)
        }
    )
    index = {} if measured is None else {h: i for i, h in enumerate(measured.hours)}
    loops = {} if measured is None else measured.loops_open
    zone_index = {} if zones is None else {h: i for i, h in enumerate(zones.hours)}
    rows: list[dict[str, Any]] = []
    for hour in day_hours(day, settings.timezone):
        night = settings.in_night(hour)
        past = hour < current
        row: dict[str, Any] = {
            "start": hour,
            "past": past,
            "price_sek_per_kwh": _hour_price(prices, hour),
            "price_quarters": _quarter_prices(prices, hour),
            "floor_c": settings.night_floor_c if night else settings.floor_c,
            "rts_measured": None,
            "rts_planned": None,
            "rts_baseline": None,
            "room_measured": None,
            "room_planned": None,
            "room_baseline": None,
            "energy_measured_kwh": None,
            "energy_planned_kwh": None,
            "energy_baseline_kwh": None,
            "saving_planned_sek": None,
            "saving_measured_sek": None,
            "flow_planned_c": None,
            "flow_baseline_c": None,
            "reason": None,
            "loops_open": {},
            "thermostats": {},
            "salon_measured": None,
            "salon_planned": None,
            "need_ground_kw": None,
            "need_salon_kw": None,
            "delivered_ground_kw": None,
            "flow_measured_c": None,
            "flow_target_c": None,
            "thermostat": None,
            "thermostat_target_c": None,
            "loops_expected": None,
        }
        if past and hour in records:
            row["thermostat"] = records[hour].thermostat
        price = row["price_sek_per_kwh"]
        if past and hour in index:
            i = index[hour]
            row["rts_measured"] = _known(measured.rts_c[i])
            row["room_measured"] = _known(measured.room_c[i])
            row["energy_measured_kwh"] = _known(
                measured.compressor_kwh[i] + measured.aux_kwh[i]
            )
            row["thermostats"] = {
                name: {
                    "room_c": _known(measured.thermostat_room_c[name][i]),
                    "set_c": _known(measured.thermostat_set_c[name][i]),
                }
                for name in measured.thermostat_room_c
            }
            row["loops_open"] = {
                name: _known(counts[i]) for name, counts in loops.items()
            }
            set_points = [
                v[i] for v in measured.thermostat_set_c.values() if np.isfinite(v[i])
            ]
            row["thermostat_target_c"] = (
                float(np.mean(set_points)) if set_points else None
            )
            t_out = float(measured.t_out_c[i])
            row["need_ground_kw"] = ground_need_kw(float(measured.room_c[i]), t_out)
            if zones is not None and hour in zone_index:
                j = zone_index[hour]
                row["salon_measured"] = _known(zones.salon_c[j])
                row["flow_measured_c"] = _known(zones.flow_c[j])
                row["flow_target_c"] = _known(zones.flow_target_c[j])
                row["need_salon_kw"] = salon_need_kw(float(zones.salon_c[j]), t_out)
                flow = float(zones.flow_c[j])
                row["delivered_ground_kw"] = delivered_ground_kw(
                    flow if np.isfinite(flow) else float(zones.flow_target_c[j]),
                    float(measured.room_c[i]),
                    measured_open_share(
                        {name: float(counts[i]) for name, counts in loops.items()},
                        {r: len(e) for r, e in settings.room_actuators.items()},
                    ),
                    1.0 - float(measured.hot_water[i]),
                    settings.floor_mix_max_c,
                )
        if past and hour in records:
            record = records[hour]
            price = record.price_sek_per_kwh
            row.update(
                rts_planned=record.rts_c,
                rts_baseline=record.baseline_rts_c,
                room_planned=record.room_c,
                room_baseline=record.baseline_room_c,
                energy_planned_kwh=record.energy_kwh,
                energy_baseline_kwh=record.baseline_energy_kwh,
                flow_planned_c=record.flow_c,
                flow_baseline_c=record.baseline_flow_c,
                reason=record.binding,
            )
        elif not past and hour in planned:
            hour_plan, baseline = planned[hour]
            price = hour_plan.price_sek_per_kwh
            row.update(
                price_sek_per_kwh=price,
                floor_c=hour_plan.floor_c,
                rts_planned=hour_plan.rts_c,
                rts_baseline=baseline.rts_c,
                room_planned=hour_plan.room_c_end + hour_plan.sun_swing_c,
                room_baseline=baseline.room_c_end + baseline.sun_swing_c,
                energy_planned_kwh=hour_plan.energy_kwh,
                energy_baseline_kwh=baseline.energy_kwh,
                flow_planned_c=hour_plan.flow_target_c,
                flow_baseline_c=baseline.flow_target_c,
                reason=hour_plan.reason.get("binding"),
                **_expected_zones(hour_plan, settings, night, salon_now_c),
            )
        row["saving_planned_sek"] = _saving(
            price, row["energy_baseline_kwh"], row["energy_planned_kwh"]
        )
        row["saving_measured_sek"] = _saving(
            price, row["energy_baseline_kwh"], row["energy_measured_kwh"]
        )
        rows.append(row)
    return {
        "day": day,
        "now": current,
        "loops_total": {
            room: len(entities) for room, entities in settings.room_actuators.items()
        },
        "hours": rows,
    }
