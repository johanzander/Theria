"""Stepwise widening of what the plan may do, driven by measured days (a trust region).

The plan starts narrow and is allowed more only after days in which nothing went wrong and the
saving was real; any sign of trouble (compressor stopped, electric heater running, room too
low) steps back and pauses the plan for a few days at the base setpoint.
"""

from dataclasses import dataclass, replace
from datetime import date, timedelta

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
    revert_days: int = 3  # days at the base setpoint after trouble
    aux_limit_kwh_per_day: float = 1.0  # electric heater energy that counts as trouble
    room_margin_bad_c: float = 0.5  # room lower than target by this much is trouble
    room_margin_widen_c: float = 0.2  # ... by this much blocks widening
    min_coverage: float = 0.5  # days with less data than this are not judged


@dataclass(frozen=True)
class StageState:
    index: int
    stage_started_on: date  # only days from here count towards the next judgement
    revert_until: date | None  # at the base setpoint until this day


@dataclass(frozen=True)
class StageDecision:
    state: StageState
    action: str  # hold | widen | shrink
    reason: str


def allowed_deviation_c(state: StageState, today: date | None = None) -> float:
    """How far from the base the setpoint may be right now (0 during a revert)."""
    if (
        today is not None
        and state.revert_until is not None
        and today < state.revert_until
    ):
        return 0.0
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
    if state.revert_until is not None and today < state.revert_until:
        return StageDecision(
            state, "hold", f"reverted to the base until {state.revert_until}"
        )
    days = [
        d
        for d in outcomes
        if state.stage_started_on <= d.day < today and d.coverage >= rules.min_coverage
    ]
    bad = [d for d in days if _trouble(d, target_c, rules)]
    if bad:
        worst = bad[0]
        return shrink_for_trouble(
            state,
            today,
            rules,
            f"trouble on {worst.day}: {_trouble(worst, target_c, rules)}",
        )
    total = float(sum(d.saving_sek for d in days))
    room_ok = all(
        not np.isfinite(d.room_min_c)
        or d.room_min_c >= target_c - rules.room_margin_widen_c
        for d in days
    )
    if (
        len(days) >= rules.min_days
        and total >= 0.0
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
    """Step back one stage and sit at the base setpoint for a few days."""
    revert_until = today + timedelta(days=rules.revert_days)
    return StageDecision(
        StageState(max(0, state.index - 1), revert_until, revert_until),
        "shrink",
        f"{reason}; base setpoint until {revert_until}",
    )


def _trouble(day: DayOutcome, target_c: float, rules: StageRules) -> str:
    """Why the day counts as trouble, or an empty string."""
    if day.compressor_stopped_hours > 0:
        return f"compressor stopped {day.compressor_stopped_hours} h"
    if day.aux_kwh > rules.aux_limit_kwh_per_day:
        return f"electric heater {day.aux_kwh:.1f} kWh"
    if (
        np.isfinite(day.room_min_c)
        and day.room_min_c < target_c - rules.room_margin_bad_c
    ):
        return f"room down to {day.room_min_c:.1f} C"
    return ""
