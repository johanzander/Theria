"""One whole local day for the charts: what was measured so far and what is planned for the rest.

Every hour of the day is listed, so the chart always spans the day from 00:00 to 24:00. Hours that
have passed carry the measured setpoint and room temperature, the hours from now on carry the
plan; prices come from the price sensor for both.
"""

from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.planner import Plan
from core.theria.plan.prices import QUARTERS_PER_HOUR, PricePeriod
from core.theria.plan.series import hour_floor
from core.theria.plan.settings import PlanSettings


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


def _hour_price(prices: dict[datetime, float], hour: datetime) -> float | None:
    quarters = [hour + timedelta(minutes=15 * i) for i in range(QUARTERS_PER_HOUR)]
    if not all(q in prices for q in quarters):
        return None
    return float(np.mean([prices[q] for q in quarters]))


def _known(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def build_day_view(
    day: date,
    now: datetime,
    settings: PlanSettings,
    periods: list[PricePeriod],
    plan: Plan | None,
    measured: OutcomeInputs | None,
) -> dict[str, Any]:
    """`measured` covers the hours before now; None when it could not be read."""
    current = hour_floor(now)
    prices = {p.start: p.price_sek_per_kwh for p in periods}
    planned = (
        {}
        if plan is None
        else {
            h.start: (h, b)
            for h, b in zip(plan.hours, plan.baseline_hours, strict=True)
        }
    )
    measured_at = (
        {}
        if measured is None
        else {
            hour: (rts, room)
            for hour, rts, room in zip(
                measured.hours, measured.rts_c, measured.room_c, strict=True
            )
        }
    )
    rows: list[dict[str, Any]] = []
    for hour in day_hours(day, settings.timezone):
        night = settings.in_night(hour)
        row: dict[str, Any] = {
            "start": hour,
            "past": hour < current,
            "price_sek_per_kwh": _hour_price(prices, hour),
            "floor_c": settings.night_floor_c if night else settings.floor_c,
            "rts_c": None,
            "room_c": None,
            "baseline_rts_c": None,
            "baseline_room_c": None,
            "extreme": False,
        }
        if row["past"] and hour in measured_at:
            rts, room = measured_at[hour]
            row["rts_c"], row["room_c"] = _known(rts), _known(room)
        elif not row["past"] and hour in planned:
            hour_plan, baseline = planned[hour]
            row.update(
                price_sek_per_kwh=hour_plan.price_sek_per_kwh,
                floor_c=hour_plan.floor_c,
                rts_c=hour_plan.rts_c,
                room_c=hour_plan.room_c_end,
                baseline_rts_c=baseline.rts_c,
                baseline_room_c=baseline.room_c_end,
                extreme=hour_plan.extreme,
            )
        rows.append(row)
    return {"day": day, "now": current, "hours": rows}
