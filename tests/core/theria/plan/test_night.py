from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.outcomes import day_outcomes

from .conftest import HORIZON
from .test_outcomes import PUMP
from .test_outcomes import make_inputs as outcome_inputs
from .test_planner import run


@pytest.fixture
def night(settings):
    """Comfort 06:00-24:00 local time; the night (00:00-06:00) may be cooler."""
    return replace(settings, night_start_hour=0, night_end_hour=6)


def test_the_night_window_follows_local_time(night):
    # Stockholm is UTC+2 in October: 22:30 UTC is 00:30 local, 04:00 UTC is 06:00 local
    assert night.in_night(datetime(2026, 10, 12, 22, 30, tzinfo=UTC))
    assert night.in_night(datetime(2026, 10, 13, 3, 59, tzinfo=UTC))
    assert not night.in_night(datetime(2026, 10, 13, 4, 0, tzinfo=UTC))
    assert not night.in_night(datetime(2026, 10, 12, 21, 59, tzinfo=UTC))


def test_a_window_across_midnight_and_an_empty_window_work(settings):
    late = replace(settings, night_start_hour=23, night_end_hour=5)
    assert late.in_night(datetime(2026, 10, 12, 21, 30, tzinfo=UTC))  # 23:30 local
    assert late.in_night(datetime(2026, 10, 13, 1, 0, tzinfo=UTC))  # 03:00 local
    assert not late.in_night(datetime(2026, 10, 13, 12, 0, tzinfo=UTC))
    assert not settings.in_night(datetime(2026, 10, 13, 1, 0, tzinfo=UTC))  # empty


def test_the_night_temperatures_must_not_exceed_the_day_ones(settings):
    with pytest.raises(ConfigurationError):
        replace(settings, night_start_hour=0, night_end_hour=6, night_target_c=20.0)
    with pytest.raises(ConfigurationError):
        replace(settings, night_start_hour=0, night_end_hour=6, night_floor_c=19.0)


def test_the_plan_reports_the_lower_floor_at_night_only(night, make_inputs):
    result = run(make_inputs(np.full(HORIZON, 1.0)), night)
    for hour in result.hours:
        end = hour.start + timedelta(
            hours=1
        )  # the room is judged at the end of the hour
        expected = night.night_floor_c if night.in_night(end) else night.floor_c
        assert hour.floor_c == expected


def test_a_cool_night_is_not_charged_as_a_comfort_shortfall(
    settings, night, make_inputs
):
    # The room starts a little under the 19.5 target. Before 06:00 that no longer costs anything.
    price = np.where(np.arange(HORIZON) < 4, 3.0, 1.0)
    without = run(make_inputs(price, room_c=19.0), settings)
    with_night = run(make_inputs(price, room_c=19.0), night)
    night_hours = [
        i
        for i, h in enumerate(with_night.hours)
        if night.in_night(h.start + timedelta(hours=1))
    ]
    assert night_hours, "the horizon covers part of the night"
    assert sum(with_night.hours[i].comfort_cost_sek for i in night_hours) < sum(
        without.hours[i].comfort_cost_sek for i in night_hours
    )
    assert with_night.objective_sek < without.objective_sek


def test_the_comfort_hours_keep_their_target(night, make_inputs):
    result = run(make_inputs(np.where(np.arange(HORIZON) < 4, 3.0, 1.0)), night)
    morning = [
        h for h in result.hours if not night.in_night(h.start + timedelta(hours=1))
    ]
    assert min(h.room_c_end for h in morning[:6]) >= night.floor_c


def test_a_cool_night_is_not_counted_as_trouble_by_the_day_judgement(night):
    inputs = outcome_inputs(18.0, 1.5, hours=48)
    inputs.room_c[:] = 19.8
    inputs.room_c[1] = 17.5  # 01:00 UTC is 03:00 local: night
    plain = day_outcomes(inputs, PUMP, rts_base_c=18.0)[0]
    aware = day_outcomes(
        inputs, PUMP, rts_base_c=18.0, comfort_hour=lambda m: not night.in_night(m)
    )[0]
    assert plain.room_min_c == pytest.approx(17.5)
    assert aware.room_min_c == pytest.approx(19.8)


def morning_peak_prices():
    """START is 02:00 local, so index i is (2 + i):00. The peak is 07-09 local: indices 5 and 6."""
    index = np.arange(HORIZON)
    return np.where((index >= 5) & (index < 7), 3.0, 1.0)


def test_the_plan_still_cuts_during_a_morning_peak_and_has_warmed_up_before_it(
    night, make_inputs
):
    result = run(make_inputs(morning_peak_prices(), room_c=19.2), night)
    rts = [h.rts_c for h in result.hours]
    assert max(rts[5:7]) < min(rts[1:5])  # warm before the peak, lower during it


def test_the_night_window_does_not_make_the_plan_flat_before_the_peak(
    night, make_inputs
):
    result = run(make_inputs(morning_peak_prices(), room_c=19.2), night)
    assert result.objective_sek < result.baseline_objective_sek - 0.1
    assert min(h.rts_c for h in result.hours[5:7]) < night.rts_base_c  # it cuts in the peak


def test_the_default_night_is_a_careful_one_until_cold_nights_have_been_measured():
    from core.theria.plan.settings import PlanSettings

    fields = PlanSettings.__dataclass_fields__
    assert fields["night_target_c"].default == 18.5
    assert fields["night_floor_c"].default == 17.5
