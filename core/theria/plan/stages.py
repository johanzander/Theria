"""Stepwise widening of what the plan may do, driven by measured days (a trust region).

The plan starts narrow and is allowed more only after days in which nothing went wrong and the
saving was real; any sign of trouble (electric heater running, room too low) steps back one
stage.
"""

from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from core.theria.plan.outcomes import DayOutcome


@dataclass(frozen=True)
class Stage:
    max_deviation_c: float  # setpoint may be this far from the base, up or down
    max_rise_per_hour_c: float  # and may be raised this fast (lowering is not limited)


STAGES = (Stage(1.0, 0.5), Stage(2.0, 1.0))  # the second is the old manual range


@dataclass(frozen=True)
class StageRules:
    min_days: int = 5  # good days needed before widening
    aux_limit_kwh_per_day: float = 1.0  # electric heater energy that counts as trouble
    room_margin_bad_c: float = 0.5  # room lower than target by this much is trouble
    room_margin_widen_c: float = 0.2  # ... by this much blocks widening
    min_coverage: float = (
        0.5  # days with less data than this do not count for the saving
    )
    min_saving_sek_per_day: float = (
        0.5  # widening needs a real saving, not days at the base
    )


@dataclass(frozen=True)
class StageState:
    index: int
    stage_started_on: date  # only days from here count towards the next judgement


@dataclass(frozen=True)
class StageDecision:
    state: StageState
    action: str  # hold | widen | shrink
    reason: str


def allowed_deviation_c(state: StageState) -> float:
    """How far from the base the setpoint may be right now."""
    return STAGES[state.index].max_deviation_c


def allowed_rise_per_hour_c(state: StageState) -> float:
    return STAGES[state.index].max_rise_per_hour_c


def evaluate_stage(
    state: StageState,
    outcomes: list[DayOutcome],
    today: date,
    target_c: float,
    rules: StageRules,
) -> StageDecision:
    """Judge the measured days since the stage started; today itself is not judged."""
    seen = [
        d
        for d in outcomes
        if state.stage_started_on <= d.day < today and d.coverage > 0
    ]
    bad = [d for d in seen if _trouble(d, target_c, rules)]
    if bad:
        worst = bad[0]
        return shrink_for_trouble(
            state,
            today,
            rules,
            f"trouble on {worst.day}: {_trouble(worst, target_c, rules)}",
        )
    days = [d for d in seen if d.coverage >= rules.min_coverage]
    total = float(sum(d.saving_sek for d in days))
    room_ok = all(
        not np.isfinite(d.room_min_c)
        or d.room_min_c >= target_c - rules.room_margin_widen_c
        for d in days
    )
    if (
        len(days) >= rules.min_days
        and total >= rules.min_saving_sek_per_day * len(days)
        and room_ok
        and state.index + 1 < len(STAGES)
    ):
        return StageDecision(
            replace(state, index=state.index + 1, stage_started_on=today),
            "widen",
            f"{len(days)} days without trouble, saving {total:.0f} SEK",
        )
    return StageDecision(
        state, "hold", f"{len(days)} judged days, saving {total:.0f} SEK"
    )


def shrink_for_trouble(
    state: StageState, today: date, rules: StageRules, reason: str
) -> StageDecision:
    """Step back one stage; only days from today count towards the next judgement."""
    index = max(0, state.index - 1)
    return StageDecision(
        StageState(index, today),
        "shrink",
        f"{reason}; stage narrowed to +/-{STAGES[index].max_deviation_c} C",
    )


def _trouble(day: DayOutcome, target_c: float, rules: StageRules) -> str:
    """Why the day counts as trouble, or an empty string."""
    if day.aux_kwh > rules.aux_limit_kwh_per_day:
        return f"electric heater {day.aux_kwh:.1f} kWh"
    if (
        np.isfinite(day.room_min_c)
        and day.room_min_c < target_c - rules.room_margin_bad_c
    ):
        return f"room down to {day.room_min_c:.1f} C"
    return ""
