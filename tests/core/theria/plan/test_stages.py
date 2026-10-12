from datetime import date, timedelta

import pytest

from core.theria.plan.outcomes import DayOutcome
from core.theria.plan.stages import (
    STAGES,
    StageRules,
    StageState,
    allowed_deviation_c,
    evaluate_stage,
)

TARGET = 19.5
START = date(2026, 11, 1)
RULES = StageRules()


def day(offset, saving=2.0, aux=0.0, room_min=19.7, coverage=1.0):
    return DayOutcome(
        day=START + timedelta(days=offset),
        heating_hours=int(24 * coverage),
        coverage=coverage,
        compressor_kwh=40.0,
        aux_kwh=aux,
        actual_cost_sek=100.0,
        baseline_cost_sek=100.0 + saving,
        saving_sek=saving,
        average_price_paid_sek_per_kwh=1.0,
        average_price_baseline_sek_per_kwh=1.0,
        mean_rts_c=18.0,
        room_min_c=room_min,
        room_mean_c=19.9,
    )


def fresh():
    return StageState(index=0, stage_started_on=START)


def run(state, days, today_offset):
    return evaluate_stage(
        state, days, START + timedelta(days=today_offset), TARGET, RULES
    )


def test_the_first_stage_is_a_narrow_one_and_the_stages_widen():
    assert STAGES[0].max_deviation_c == 1.0 and STAGES[0].max_rise_per_hour_c == 0.5
    assert STAGES[1].max_deviation_c == 2.0 and STAGES[1].max_rise_per_hour_c == 1.0
    assert allowed_deviation_c(fresh()) == 1.0


def test_it_holds_until_enough_good_days_have_passed():
    decision = run(fresh(), [day(i) for i in range(4)], today_offset=4)
    assert decision.action == "hold" and decision.state.index == 0


def test_five_good_days_with_a_positive_total_widen_one_stage():
    decision = run(fresh(), [day(i) for i in range(5)], today_offset=5)
    assert decision.action == "widen" and decision.state.index == 1
    assert decision.state.stage_started_on == START + timedelta(days=5)
    assert "5 days" in decision.reason


def test_a_negative_total_saving_stops_the_widening_but_is_not_a_reason_to_shrink():
    days = [day(i, saving=-1.0) for i in range(6)]
    decision = run(fresh(), days, today_offset=6)
    assert decision.action == "hold" and decision.state.index == 0


def test_there_is_no_stage_beyond_the_last():
    state = StageState(index=len(STAGES) - 1, stage_started_on=START)
    decision = run(state, [day(i) for i in range(8)], today_offset=8)
    assert decision.action == "hold" and decision.state.index == len(STAGES) - 1


@pytest.mark.parametrize(
    "bad",
    [
        {"aux": 2.5},
        {"room_min": 18.9},
    ],
)
def test_a_bad_day_shrinks_the_stage_and_judging_starts_over(bad):
    state = StageState(index=1, stage_started_on=START)
    days = [day(0), day(1, **bad)]
    decision = run(state, days, today_offset=2)
    assert decision.action == "shrink" and decision.state.index == 0
    assert decision.state.stage_started_on == START + timedelta(days=2)


def test_at_the_narrowest_stage_a_bad_day_only_starts_the_count_over():
    decision = run(fresh(), [day(0, aux=2.5)], today_offset=1)
    assert decision.action == "shrink" and decision.state.index == 0
    assert decision.state.stage_started_on == START + timedelta(days=1)


def test_a_slightly_low_room_blocks_widening_without_shrinking():
    days = [day(i, room_min=19.1 if i == 2 else 19.7) for i in range(6)]
    decision = run(fresh(), days, today_offset=6)
    assert decision.action == "hold"


def test_days_before_the_stage_started_are_ignored():
    state = StageState(index=0, stage_started_on=START + timedelta(days=3))
    days = [
        day(0, aux=5),
        day(1, aux=5),
        day(3),
        day(4),
    ]
    decision = run(state, days, today_offset=6)
    assert (
        decision.action == "hold"
    )  # two good days only; the earlier bad ones do not count


def test_days_without_a_real_saving_do_not_widen_the_stage():
    days = [day(i, saving=0.0) for i in range(8)]  # paused, fallback or at the base
    decision = run(fresh(), days, today_offset=8)
    assert decision.action == "hold" and decision.state.index == 0


def test_trouble_is_judged_even_on_days_with_little_normal_heating_data():
    days = [day(i, coverage=0.2, aux=3.0) for i in range(1)]
    decision = run(fresh(), days, today_offset=1)
    assert decision.action == "shrink"
