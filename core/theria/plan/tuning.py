"""Advice on the settings the plan never changes by itself, from what the days have shown.

Nothing here is applied: each recommendation names a setting, its current value, what the data
says and, when the data speaks clearly, a value to consider. Johan decides.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

import numpy as np

from core.theria.plan.night_drift import NightDrift
from core.theria.plan.outcomes import DayOutcome
from core.theria.plan.settings import PlanSettings
from core.theria.plan.stages import STAGES, StageRules, StageState

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


def _stage(
    state: StageState, days: list[DayOutcome], rules: StageRules, today: date
) -> Recommendation:
    title = "Stage (how far the plan may move the setpoint)"
    current = f"± {STAGES[state.index].max_deviation_c} °C"
    if state.index + 1 >= len(STAGES):
        return Recommendation(
            "stage",
            title,
            current,
            "widest stage",
            None,
            "There is no wider stage.",
            "ok",
        )
    judged = [
        d
        for d in days
        if state.stage_started_on <= d.day < today and d.coverage >= rules.min_coverage
    ]
    missing = max(0, rules.min_days - len(judged))
    saving = float(sum(d.saving_sek for d in judged))
    observed = (
        f"{len(judged)} judged days, saving {saving:.0f} SEK; {missing} more days needed"
        if missing
        else f"{len(judged)} judged days, saving {saving:.0f} SEK"
    )
    return Recommendation(
        "stage",
        title,
        current,
        observed,
        None,
        f"It widens by itself after {rules.min_days} days without trouble and a saving of at "
        f"least {rules.min_saving_sek_per_day} SEK per day.",
        "ok",
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
    stage: StageState,
    night: NightDrift | None,
    rules: StageRules,
    today: date,
) -> list[Recommendation]:
    stage_note = (
        "The stage limits how far from the base the setpoint may go, so widening the stage "
        "comes before changing this limit."
    )
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
            stage_note,
        ),
        _share_rec(
            "rts_max_c",
            "Highest setpoint",
            f"{settings.rts_max_c:.1f} °C",
            plan_hours,
            "rts_max",
            LIMIT_SHARE,
            stage_note,
        ),
        _stage(stage, days, rules, today),
        _night_model(night),
    ]
