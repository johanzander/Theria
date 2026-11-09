from datetime import UTC, datetime

import numpy as np

from core.theria.plan.shift_rules import (
    SalonStatus,
    ShiftReadings,
    ShiftWindow,
    balanced,
    guard_reason,
)

END = datetime(2026, 11, 9, 19, 0, tzinfo=UTC)


def readings(room=19.5, salon=None):
    return ShiftReadings(room_c=room, salon=salon)


def test_more_sold_than_bought_gives_up_the_cheapest_sells():
    adjust = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    prices = np.array([1.0, 2.0, 9.0, 8.0, 7.0, 6.0])
    assert list(balanced(adjust, prices)) == [1, 1, -1, -1, 0, 0]


def test_more_bought_than_sold_gives_up_the_dearest_buys():
    adjust = np.array([1.0, 1.0, 1.0, -1.0])
    prices = np.array([1.0, 3.0, 2.0, 9.0])
    assert list(balanced(adjust, prices)) == [1, 0, 0, -1]


def test_a_small_imbalance_is_left_alone():
    adjust = np.array([1.0] * 10 + [-1.0] * 11)
    assert list(balanced(adjust, np.arange(21.0))) == list(adjust)


def test_no_shift_is_balanced():
    assert list(balanced(np.zeros(3), np.ones(3))) == [0, 0, 0]


def test_a_room_under_its_floor_stops_a_sell():
    assert guard_reason(-1.0, readings(room=18.4)) is None


def test_a_room_at_its_floor_does_not():
    assert guard_reason(-1.0, readings(room=18.5)) is None


def test_a_cold_room_does_not_stop_a_buy():
    assert guard_reason(1.0, readings(room=18.0)) is None


def test_a_salon_under_its_floor_stops_a_sell():
    salon = SalonStatus(mean_c=18.9, floor_c=19.0)
    assert guard_reason(-1.0, readings(salon=salon)) == "salon_floor"


def test_a_salon_nobody_can_read_stops_a_sell():
    salon = SalonStatus(mean_c=None, floor_c=19.0)
    assert guard_reason(-1.0, readings(salon=salon)) == "salon_unknown"


def test_the_salon_is_not_checked_when_there_is_none():
    assert guard_reason(-1.0, readings(salon=None)) is None


def test_a_buy_is_never_stopped():
    salon = SalonStatus(mean_c=None, floor_c=19.0)
    assert guard_reason(1.0, readings(room=17.0, salon=salon)) is None


def test_a_window_survives_being_written_and_read():
    window = ShiftWindow(END, -1.0, "salon_floor")
    assert ShiftWindow.from_dict(window.to_dict()) == window


def test_the_readings_survive_being_written_and_read():
    shift = ShiftReadings(19.0, SalonStatus(None, 19.0), ShiftWindow(END, 1.0, None))
    assert ShiftReadings.from_dict(shift.to_dict()) == shift
