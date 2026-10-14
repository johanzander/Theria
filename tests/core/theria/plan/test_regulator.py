from datetime import UTC, datetime, timedelta

import pytest

from core.theria.plan.regulator import (
    HYSTERESIS_C,
    MAX_RISE_C,
    Regulator,
    RoomRegulator,
)
from core.theria.plan.settings import RoomControl

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
LIMIT = 22.0


def feed(regulator, temps, valves=None, minutes=15):
    """Feed one temperature per step; returns the last status."""
    status = None
    for k, temp in enumerate(temps):
        valve = None if valves is None else valves[k]
        status = regulator.update(NOW + timedelta(minutes=minutes * k), temp, valve)
    return status


def test_a_room_well_below_its_limit_stays_open():
    status = feed(RoomRegulator("a", LIMIT), [20.0, 20.1, 20.2])
    assert status.state == "open"


def test_a_rising_room_that_reaches_the_close_level_is_closed():
    status = feed(RoomRegulator("a", LIMIT), [21.5, 21.8, 21.95])
    assert status.state == "closed"


def test_a_room_that_hovers_at_the_level_but_is_falling_is_not_closed():
    status = feed(RoomRegulator("a", LIMIT), [21.95, 21.92, 21.9])
    assert status.state == "open"


def test_a_closed_room_reopens_only_after_falling_a_hysteresis_below_the_level():
    regulator = RoomRegulator("a", LIMIT)
    feed(regulator, [21.5, 21.9, 22.1])
    assert regulator.update(NOW + timedelta(hours=1), 21.8, None).state == "closed"
    status = regulator.update(
        NOW + timedelta(hours=2), LIMIT - 0.1 - HYSTERESIS_C - 0.05, None
    )
    assert status.state == "open"


def test_a_room_without_a_reading_is_unknown_and_not_closed():
    status = RoomRegulator("a", LIMIT).update(NOW, None, None)
    assert status.state == "unknown" and status.temperature_c is None


def test_the_rise_after_the_valves_close_is_learned_and_closes_earlier_next_time():
    regulator = RoomRegulator("a", LIMIT)
    # valves close (fraction 0) at 21.9; the floor keeps releasing heat: peak 22.6, then it falls
    temps = [21.5, 21.9, 22.2, 22.5, 22.6, 22.4, 22.2]
    valves = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    feed(regulator, temps, valves)
    assert regulator.rise_c == pytest.approx(0.7)
    assert regulator.update(NOW + timedelta(hours=3), 21.0, 1.0).close_at_c == pytest.approx(
        LIMIT - 0.7
    )


def test_the_learned_rise_is_bounded_and_averaged_with_earlier_ones():
    regulator = RoomRegulator("a", LIMIT, rise_c=0.4, events=1)
    feed(regulator, [21.9, 22.0, 22.1, 22.3, 22.4, 22.0], [0.0] * 6)
    assert regulator.rise_c == pytest.approx(0.5 * 0.4 + 0.5 * 0.5)
    huge = RoomRegulator("b", LIMIT)
    feed(huge, [21.9, 25.9, 26.0, 20.0], [0.0] * 4)
    assert huge.rise_c <= MAX_RISE_C


def test_without_actuator_readings_nothing_is_learned():
    regulator = RoomRegulator("a", LIMIT)
    feed(regulator, [21.9, 22.5, 22.7, 22.0], None)
    assert regulator.rise_c == 0.0


def test_the_state_survives_a_restart():
    regulator = RoomRegulator("a", LIMIT, rise_c=0.6, events=3)
    again = RoomRegulator("a", LIMIT)
    again.load_state(regulator.export_state())
    assert again.rise_c == 0.6


# ---- several rooms

CONTROLS = (
    RoomControl("vardagsrum", "climate.lk_v", ("sensor.v1", "sensor.v2")),
    RoomControl("bibliotek", "climate.lk_b", ("sensor.b1",)),
)


def states(**values):
    def read(entity_id):
        return values[entity_id]

    return read


def test_the_room_temperature_is_the_mean_of_its_valid_sensors():
    regulator = Regulator(CONTROLS, LIMIT, actuators={})
    read = states(**{"sensor.v1": "20.0", "sensor.v2": "unavailable", "sensor.b1": "19.0"})
    statuses = {s.room: s for s in regulator.step(NOW, read)}
    assert statuses["vardagsrum"].temperature_c == 20.0  # the unavailable one is dropped
    read = states(**{"sensor.v1": "20.0", "sensor.v2": "21.0", "sensor.b1": "19.0"})
    statuses = {s.room: s for s in regulator.step(NOW + timedelta(minutes=15), read)}
    assert statuses["vardagsrum"].temperature_c == 20.5


def test_only_the_room_over_its_limit_is_closed_and_the_other_keeps_heating():
    regulator = Regulator(CONTROLS, LIMIT, actuators={})
    read = states(**{"sensor.v1": "22.4", "sensor.v2": "22.2", "sensor.b1": "20.0"})
    regulator.step(NOW, read)
    statuses = {s.room: s for s in regulator.step(NOW + timedelta(minutes=15), read)}
    assert statuses["vardagsrum"].state == "closed"
    assert statuses["bibliotek"].state == "open"
    holds = regulator.holds(list(statuses.values()))
    assert set(holds) == {"climate.lk_v"}
    assert holds["climate.lk_v"] <= 22.3 - 0.5 and holds["climate.lk_v"] * 2 == int(holds["climate.lk_v"] * 2)


def test_the_valve_state_of_a_room_is_read_from_its_actuators():
    regulator = Regulator(CONTROLS, LIMIT, actuators={"bibliotek": ("binary_sensor.a1", "binary_sensor.a2")})
    read = states(
        **{
            "sensor.v1": "20", "sensor.v2": "20", "sensor.b1": "19",
            "binary_sensor.a1": "on", "binary_sensor.a2": "off",
        }
    )
    statuses = {s.room: s for s in regulator.step(NOW, read)}
    assert statuses["bibliotek"].valve_fraction == 0.5
    assert statuses["vardagsrum"].valve_fraction is None
