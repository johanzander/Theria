from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pytest

from core.theria.plan.morning import MorningRules, update_margin
from core.theria.plan.outcomes import day_outcomes

from .days import day
from .test_outcomes import PUMP
from .test_outcomes import make_inputs as outcome_inputs

RULES = MorningRules()
TARGET = 19.5
TODAY = date(2026, 11, 10)


def morning(offset, room):
    return replace(
        day(offset),
        day=date(2026, 11, 1) + timedelta(days=offset),
        room_at_morning_c=room,
    )


def run(margin, rooms, since=date(2026, 11, 1), cap=1.0):
    days = [morning(i, r) for i, r in enumerate(rooms)]
    return update_margin(margin, days, TARGET, TODAY, since, RULES, cap)


def test_a_morning_that_was_too_cold_raises_the_margin_for_the_next_night():
    decision = run(0.0, [19.6, 19.0])  # 0.5 C short on the last day
    assert decision.action == "raise" and decision.margin_c == pytest.approx(0.25)
    assert decision.changed_on == TODAY and "19.0" in decision.reason


def test_the_margin_never_exceeds_what_would_cancel_the_night_window():
    assert run(0.9, [18.9]).margin_c == pytest.approx(1.0)
    assert run(1.0, [18.0], cap=1.0).margin_c == pytest.approx(1.0)
    assert run(0.4, [18.0], cap=0.5).margin_c == pytest.approx(0.5)


def test_three_comfortable_mornings_in_a_row_lower_it_again():
    decision = run(0.5, [19.6, 19.7, 19.8])
    assert decision.action == "lower" and decision.margin_c == pytest.approx(0.4)


def test_it_does_not_go_below_zero():
    assert run(0.05, [19.6, 19.7, 19.8]).margin_c == pytest.approx(0.0)


def test_it_waits_for_evidence_before_lowering():
    assert run(0.5, [19.6, 19.7]).action == "hold"


def test_days_before_the_last_change_are_not_counted_twice():
    decision = update_margin(
        0.5,
        [morning(0, 18.5), morning(1, 19.6), morning(2, 19.6)],
        TARGET,
        TODAY,
        since=date(2026, 11, 3),  # the margin was changed after the cold morning
        rules=RULES,
        cap_c=1.0,
    )
    assert decision.action == "hold"


def test_mornings_without_a_measurement_say_nothing():
    assert run(0.3, [float("nan"), float("nan")]).action == "hold"


def test_the_day_outcome_records_the_room_at_the_start_of_comfort():
    inputs = outcome_inputs(18.0, 1.5, hours=48)
    inputs.room_c[:] = 19.8
    inputs.room_c[4] = 18.7  # 04:00 UTC = 06:00 local
    outcome = day_outcomes(
        inputs, PUMP, rts_base_c=18.0, morning_hour=lambda m: m.hour == 4
    )[0]
    assert outcome.room_at_morning_c == pytest.approx(18.7)
    assert np.isnan(day_outcomes(inputs, PUMP, rts_base_c=18.0)[0].room_at_morning_c)
