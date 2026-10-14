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
    flow_target_c: float


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


def _floor_cap(settings: PlanSettings, hours: Sequence[PlannedHour]) -> Recommendation:
    cap = settings.floor_mix_max_c
    share = (
        sum(1 for h in hours if h.flow_target_c > cap) / len(hours) if hours else 0.0
    )
    return Recommendation(
        "floor_mix_max_c",
        "Floor mixing valve limit",
        f"{cap:.0f} °C",
        f"the pump asks for more than that in {share * 100:.0f} % of the planned hours",
        None,
        "The floor receives only the limit whatever the pump asks for, so lowering the pump setpoint "
        "costs the floor nothing in those hours (the plan does that); check that the limit is what "
        "the mixing valves are set to.",
        "consider" if share >= CAP_SHARE else "ok",
    )


def _limit_rec(
    key: str,
    title: str,
    current: str,
    hours: Sequence[PlannedHour],
    binding: str,
    authority: float,
) -> Recommendation:
    if authority < 1.0:
        return Recommendation(
            key,
            title,
            current,
            f"the plan has {authority * 100:.0f} % of the range until the model has shown it predicts well",
            None,
            "The range the plan may use is widened as the model proves itself, so a limit is not "
            "judged until it has the whole range.",
            "no data",
        )
    return _share_rec(
        key,
        title,
        current,
        hours,
        binding,
        LIMIT_SHARE,
        "A wider range lets the plan pre-heat more; watch the electric heater when raising it.",
    )


def build_recommendations(
    settings: PlanSettings,
    days: list[DayOutcome],
    plan_hours: Sequence[PlannedHour],
    night: NightDrift | None,
    authority: float = 1.0,
) -> list[Recommendation]:
    return [
        _floor(settings, days),
        _floor_cap(settings, plan_hours),
        _limit_rec(
            "rts_min_c", "Lowest setpoint", f"{settings.rts_min_c:.1f} °C", plan_hours, "rts_min", authority
        ),
        _limit_rec(
            "rts_max_c", "Highest setpoint", f"{settings.rts_max_c:.1f} °C", plan_hours, "rts_max", authority
        ),
        _night_model(night),
    ]
