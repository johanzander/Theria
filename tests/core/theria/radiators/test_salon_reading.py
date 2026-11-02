from datetime import UTC, datetime

import pytest

from core.theria.radiators.salon_reading import read_salon
from core.theria.sensor_guard import STATUS_FAULT, STATUS_OK, Issue, SensorReport

NOW = datetime(2026, 11, 1, 12, tzinfo=UTC)
THERMOSTAT_SENSORS = {"sensor.v", "sensor.k"}


def report(entity, value, usable=True, issue=None):
    return SensorReport(
        name=entity,
        room="salon",
        temperature_entity=entity,
        status=STATUS_OK if usable else STATUS_FAULT,
        issues=() if issue is None else (Issue("silent", STATUS_FAULT, issue),),
        value_c=value,
        age_minutes=5.0,
        battery_percent=None,
        signal_dbm=None,
        offset_c=None,
        reference=entity == "sensor.v",
        checked_at=NOW,
        usable=usable,
    )


FOUR = [
    report("sensor.v", 20.5),
    report("sensor.k", 20.0),
    report("sensor.floor", 19.7),
    report("sensor.ceiling", 19.1),
]


def test_mean_is_the_plain_mean_of_usable_points():
    reading = read_salon(FOUR, THERMOSTAT_SENSORS)
    assert reading.mean_c == pytest.approx(19.825)
    assert not reading.unknown
    assert set(reading.used) == {e.temperature_entity for e in FOUR}


def test_a_silent_point_is_left_out_and_named():
    reports = [*FOUR[:3], report("sensor.ceiling", 19.1, usable=False, issue="silent")]
    reading = read_salon(reports, THERMOSTAT_SENSORS)
    assert reading.mean_c == pytest.approx((20.5 + 20.0 + 19.7) / 3)
    assert "sensor.ceiling" in reading.skipped
    assert "silent" in reading.skipped["sensor.ceiling"]


def test_one_usable_point_is_unknown():
    reports = [
        FOUR[0],
        *[report(e, None, usable=False) for e in ("sensor.k", "x", "y")],
    ]
    reading = read_salon(reports, THERMOSTAT_SENSORS)
    assert reading.unknown
    assert reading.mean_c is None


def test_no_points_is_unknown():
    reading = read_salon([], THERMOSTAT_SENSORS)
    assert reading.unknown
    assert reading.mean_c is None
    assert reading.thermostat_mean_c is None


def test_thermostat_mean_uses_only_the_thermostat_sensors():
    reading = read_salon(FOUR, THERMOSTAT_SENSORS)
    assert reading.thermostat_mean_c == pytest.approx(20.25)
