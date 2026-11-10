from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.shift_rules import (
    CLOSING,
    NORMAL,
    OPENING,
    ShiftWindow,
    thermostat_modes,
    thermostat_target,
)

T0 = datetime(2026, 11, 9, 12, tzinfo=UTC)


def hours(n):
    return [T0 + timedelta(hours=i) for i in range(n)]


def modes(adjust, window=None, gap=2):
    return thermostat_modes(np.array(adjust, float), hours(len(adjust)), window, gap)


def test_a_sell_closes_and_other_hours_are_normal():
    assert modes([0, -1, -1, 0]) == [NORMAL, CLOSING, CLOSING, NORMAL]


def test_a_buy_right_before_a_sell_opens():
    assert modes([1, 1, -1]) == [OPENING, OPENING, CLOSING]


def test_a_buy_within_the_gap_opens():
    assert modes([1, 0, 0, -1], gap=2) == [OPENING, NORMAL, NORMAL, CLOSING]


def test_a_buy_further_from_its_sell_leaves_the_thermostat_alone():
    assert modes([1, 0, 0, 0, -1], gap=2) == [NORMAL, NORMAL, NORMAL, NORMAL, CLOSING]


def test_a_buy_with_no_sell_after_it_leaves_the_thermostat_alone():
    assert modes([0, 1, 1, 0]) == [NORMAL, NORMAL, NORMAL, NORMAL]


def test_a_sell_ended_by_a_guard_keeps_closing_to_the_end_of_the_window():
    window = ShiftWindow(T0 + timedelta(hours=2), -1.0, "salon_floor")
    assert modes([0, 0, 0], window) == [CLOSING, CLOSING, NORMAL]


def test_a_buy_window_does_not_close():
    window = ShiftWindow(T0 + timedelta(hours=2), 1.0)
    assert modes([1, 1, 0], window) == [NORMAL, NORMAL, NORMAL]


@pytest.fixture
def lk(settings):
    return replace(settings)


def test_closing_is_the_lower_limit_plus_the_offset_by_day(lk):
    assert thermostat_target(CLOSING, False, 19.5, lk) == 19.0


def test_closing_is_the_night_limit_plus_the_offset_at_night(lk):
    assert thermostat_target(CLOSING, True, 19.5, lk) == 18.0


def test_opening_is_the_upper_limit(lk):
    assert thermostat_target(OPENING, False, 19.5, lk) == 20.5


def test_a_room_close_to_the_upper_limit_is_not_opened(lk):
    assert thermostat_target(OPENING, False, 20.1, lk) == lk.lk_base_c


def test_a_room_that_cannot_be_read_is_not_opened(lk):
    assert thermostat_target(OPENING, False, None, lk) == lk.lk_base_c


def test_normal_is_the_base(lk):
    assert thermostat_target(NORMAL, False, 19.5, lk) == 19.5
