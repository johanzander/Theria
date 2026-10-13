"""Advice on the settings the plan never changes by itself, from what the days have shown.

Nothing here is applied: each recommendation names a setting, its current value, what the data
says and, when the data speaks clearly, a value to consider. Johan decides.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from core.theria.plan.night_drift import NightDrift
from core.theria.plan.outcomes import DayOutcome
from core.theria.plan.settings import PlanSettings

MIN_DAYS = 7  # days with enough data before the room's lows are trusted
FLOOR_SLACK_C = 1.5  # a floor the room stays this far above is not doing anything
FLOOR_STEP_C = 0.5
LIMIT_SHARE = (
    0.5  # share of the planned hours held by a setpoint limit worth mentioning
)
CAP_SHARE = 0.2  # the same for the flow cap
MIN_NIGHTS = 60


class PlannedHour(Protocol):
    reason: dict[str, Any]


@dataclass(frozen=True)
class Recommendation:
    key: str
    title: str
    current: str
    observed: str
    suggestion: str | None
    reason: str
    status: str  # ok | consider | no data


def _share(hours: Sequence[PlannedHour], binding: str) -> float:
    if not hours:
        return 0.0
    return sum(1 for h in hours if h.reason.get("binding") == binding) / len(hours)


def _floor(s: PlanSettings, days: list[DayOutcome]) -> Recommendation:
    current = f"{s.floor_c:.1f} °C"
    rooms = [d.room_min_c for d in days if np.isfinite(d.room_min_c)]
    title = "Comfort floor (daytime)"
    if len(rooms) < MIN_DAYS:
        return Recommendation(
            "floor_c",
            title,
            current,
            f"{len(rooms)} days with data, {MIN_DAYS} needed",
            None,
            "The floor is only judged against days with enough measurements.",
            "no data",
        )
    lowest = float(min(rooms))
    observed = f"the room's lowest daytime temperature in {len(rooms)} days was {lowest:.1f} °C"
    if lowest - s.floor_c < FLOOR_SLACK_C:
        return Recommendation(
            "floor_c",
            title,
            current,
            observed,
            None,
            "The room comes close to the floor, so it is doing its job.",
            "ok",
        )
    suggested = max(s.extreme_floor_c, s.floor_c - FLOOR_STEP_C)
    return Recommendation(
        "floor_c",
        title,
        current,
        observed,
        f"{suggested:.1f} °C",
        "The room never came near the floor; a lower floor lets the plan coast further "
        "in expensive hours. This trades comfort margin for saving, so it is your call.",
        "consider",
    )


def _share_rec(
    key: str,
    title: str,
    current: str,
    hours: Sequence[PlannedHour],
    binding: str,
    limit_share: float,
    reason: str,
) -> Recommendation:
    share = _share(hours, binding)
    observed = f"{share * 100:.0f} % of the planned hours are held by it"
    return Recommendation(
        key,
        title,
        current,
        observed,
        None,
        reason,
        "consider" if share >= limit_share else "ok",
    )


def _night_model(night: NightDrift | None) -> Recommendation:
    title = "Night cooling model"
    if night is None:
        return Recommendation(
            "night_model",
            title,
            "not learned",
            "no night model yet",
            None,
            "It needs steady nights over a range of outdoor temperatures.",
            "no data",
        )
    observed = f"{night.nights} nights, error {night.rmse_c:.2f} °C"
    if night.nights < MIN_NIGHTS:
        return Recommendation(
            "night_model",
            title,
            f"{night.nights} nights",
            observed,
            None,
            "Few nights: the cold end of the model is uncertain, so the night margin "
            "does most of the work for now.",
            "consider",
        )
    return Recommendation(
        "night_model",
        title,
        f"{night.nights} nights",
        observed,
        None,
        "Enough nights to trust the model.",
        "ok",
    )


def build_recommendations(
    settings: PlanSettings,
    days: list[DayOutcome],
    plan_hours: Sequence[PlannedHour],
    night: NightDrift | None,
) -> list[Recommendation]:
    heater_note = "A wider range lets the plan pre-heat more; watch the electric heater when raising it."
    return [
        _floor(settings, days),
        _share_rec(
            "max_flow_c",
            "Maximum flow temperature",
            f"{settings.max_flow_c:.0f} °C",
            plan_hours,
            "flow_cap",
            CAP_SHARE,
            "The 45 °C is what the floor was designed for; raising it is a hardware "
            "and parquet question, not a tuning one.",
        ),
        _share_rec(
            "rts_min_c",
            "Lowest setpoint",
            f"{settings.rts_min_c:.1f} °C",
            plan_hours,
            "rts_min",
            LIMIT_SHARE,
            heater_note,
        ),
        _share_rec(
            "rts_max_c",
            "Highest setpoint",
            f"{settings.rts_max_c:.1f} °C",
            plan_hours,
            "rts_max",
            LIMIT_SHARE,
            heater_note,
        ),
        _night_model(night),
    ]
