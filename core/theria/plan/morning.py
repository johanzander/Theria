"""Learning how warm the night must be kept, from how warm the mornings turned out.

The night window lets the room be cooler until comfort is due (06:00). Whether the room is
back at the target by then depends on how cold the night was and how slowly the floor
responds, which the models know only roughly. So the plan is closed with feedback: every day
the room at 06:00 is compared with the target. A morning that was too cold raises a margin
that lifts the night's target and floor (the plan then keeps the room warmer, i.e. starts
heating earlier); several comfortable mornings in a row lower it again. The margin is the one
number that is learned from the outcome rather than from the models.
"""

from dataclasses import dataclass
from datetime import date

import numpy as np

from core.theria.plan.outcomes import DayOutcome


@dataclass(frozen=True)
class MorningRules:
    shortfall_trigger_c: float = (
        0.3  # a morning this far under the target raises the margin
    )
    step_up_c: float = 0.25
    step_down_c: float = 0.1
    days_to_lower: int = 3  # comfortable mornings in a row before the margin is lowered


@dataclass(frozen=True)
class MarginDecision:
    margin_c: float
    action: str  # raise | lower | hold
    reason: str
    changed_on: date | None  # set when the margin changed


def update_margin(
    margin_c: float,
    outcomes: list[DayOutcome],
    target_c: float,
    today: date,
    since: date,
    rules: MorningRules,
    cap_c: float,
) -> MarginDecision:
    """Judge the mornings since the margin was last changed (today is not judged)."""
    mornings = [
        (d.day, d.room_at_morning_c)
        for d in outcomes
        if since <= d.day < today and np.isfinite(d.room_at_morning_c)
    ]
    if not mornings:
        return MarginDecision(margin_c, "hold", "no measured mornings yet", None)
    worst_day, worst_room = min(mornings, key=lambda m: m[1])
    if target_c - worst_room > rules.shortfall_trigger_c:
        raised = min(cap_c, margin_c + rules.step_up_c)
        return MarginDecision(
            raised,
            "raise" if raised > margin_c else "hold",
            f"the room was {worst_room:.1f} C at the start of comfort on {worst_day} "
            f"(target {target_c:.1f} C): night margin {margin_c:.2f} -> {raised:.2f} C",
            today if raised > margin_c else None,
        )
    if len(mornings) >= rules.days_to_lower and all(
        room >= target_c for _, room in mornings
    ):
        lowered = max(0.0, margin_c - rules.step_down_c)
        return MarginDecision(
            lowered,
            "lower" if lowered < margin_c else "hold",
            f"{len(mornings)} comfortable mornings in a row: night margin "
            f"{margin_c:.2f} -> {lowered:.2f} C",
            today if lowered < margin_c else None,
        )
    return MarginDecision(margin_c, "hold", f"{len(mornings)} mornings judged", None)
