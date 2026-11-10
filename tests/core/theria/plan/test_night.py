from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.outcomes import day_outcomes

from .conftest import HORIZON
from .test_outcomes import PUMP
from .test_outcomes import make_inputs as outcome_inputs


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


def test_the_default_night_is_a_careful_one_until_cold_nights_have_been_measured():
    from core.theria.plan.settings import PlanSettings

    fields = PlanSettings.__dataclass_fields__
    assert fields["night_target_c"].default == 18.5
    assert fields["night_floor_c"].default == 17.5
