import os
from datetime import UTC, datetime, time

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.radiators.settings import (
    DaySchedule,
    GuestRoom,
    RadiatorPrefs,
    load_radiator_settings,
)

TZ = "Europe/Stockholm"
CONFIG = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "config.yaml")


def local(day: int, hour: int, minute: int = 0, month: int = 11) -> datetime:
    """A moment given as Stockholm wall-clock time on 2026-<month>-<day> (winter time, UTC+1)."""
    return datetime(2026, month, day, hour - 1, minute, tzinfo=UTC)


def test_defaults_match_the_spec():
    prefs = RadiatorPrefs()
    assert prefs.salon_schedule["mon"] == DaySchedule(time(9), time(19))
    assert prefs.salon_schedule["fri"] == DaySchedule(time(9), time(19))
    assert prefs.salon_schedule["sat"] == DaySchedule(time(10), time(16))
    assert prefs.salon_schedule["sun"] == DaySchedule(time(10), time(16))
    assert prefs.salon_open_target_c == 20.0
    assert prefs.salon_open_floor_c == 19.0
    assert prefs.salon_closed_floor_c == 17.0
    assert prefs.salon_preheat_c == 21.0
    assert prefs.salon_margin_c == 0.3
    assert prefs.salon_max_setpoint_c == 22.0
    assert prefs.guest_band_c == 1.5
    assert prefs.mode == "shadow"


def test_is_open_at_the_edges():
    prefs = RadiatorPrefs()
    # 2026-11-02 is a Monday, 2026-11-01 a Sunday
    assert prefs.is_open(local(2, 9), TZ)
    assert prefs.is_open(local(2, 18, 59), TZ)
    assert not prefs.is_open(local(2, 19), TZ)
    assert not prefs.is_open(local(2, 8, 59), TZ)
    assert prefs.is_open(local(1, 15, 59), TZ)
    assert not prefs.is_open(local(1, 16), TZ)


def test_is_open_uses_local_time_across_dst():
    prefs = RadiatorPrefs()
    # Monday 2026-10-26 09:00 local is 08:00 UTC (winter time); Sunday 2026-10-25 had summer time until 03:00
    assert prefs.is_open(datetime(2026, 10, 26, 8, 0, tzinfo=UTC), TZ)
    assert not prefs.is_open(datetime(2026, 10, 26, 7, 59, tzinfo=UTC), TZ)
    # Saturday 2026-10-24 10:00 local is 08:00 UTC (summer time, UTC+2)
    assert prefs.is_open(datetime(2026, 10, 24, 8, 0, tzinfo=UTC), TZ)
    assert not prefs.is_open(datetime(2026, 10, 24, 7, 59, tzinfo=UTC), TZ)


def test_a_day_set_to_none_is_closed():
    schedule = {**RadiatorPrefs().salon_schedule, "sun": None}
    prefs = RadiatorPrefs(salon_schedule=schedule)
    assert not prefs.is_open(local(1, 12), TZ)


@pytest.mark.parametrize(
    "change",
    [
        {"salon_preheat_c": 19.2},  # below open_target + margin
        {"salon_max_setpoint_c": 19.5},  # below preheat
        {"guest_band_c": 0.0},
        {"mode": "active"},
        {"salon_closed_floor_c": 19.5},  # above the open target
    ],
)
def test_invalid_preferences_raise_configuration_error(change):
    with pytest.raises(ConfigurationError):
        RadiatorPrefs(**change)


def test_a_schedule_with_close_before_open_is_invalid():
    with pytest.raises(ConfigurationError):
        DaySchedule(time(19), time(9))


def test_prefs_round_trip_through_dict():
    prefs = RadiatorPrefs(
        salon_open_target_c=19.5,
        guest_rooms=(GuestRoom("climate.bedroom", 18.0, 2.0),),
        salon_schedule={**RadiatorPrefs().salon_schedule, "sun": None},
    )
    assert RadiatorPrefs.from_dict(prefs.to_dict()) == prefs


def test_unknown_keys_raise():
    with pytest.raises(ConfigurationError):
        RadiatorPrefs.from_dict({"salon_color": 1})


def test_config_yaml_radiators_section_loads():
    settings = load_radiator_settings("/nonexistent/options.json", CONFIG)
    assert settings is not None
    assert settings.salon_thermostats == ("climate.vantsidan", "climate.klippsidan")
    assert "sensor.schamponeringstolen_temperature" in settings.salon_sensors
    assert settings.defaults.mode == "shadow"
    assert {r.entity for r in settings.defaults.guest_rooms} == {
        "climate.bedroom",
        "climate.hallway",
        "climate.kitchen",
        "climate.tv_room_gh",
        "climate.storage_room",
    }
    assert len(settings.salon_sensor_guard.sensors) == 4


def test_an_hour_overlapping_the_opening_time_counts_as_open():
    schedule = {
        **RadiatorPrefs().salon_schedule,
        "mon": DaySchedule(time(9, 30), time(18, 30)),
    }
    prefs = RadiatorPrefs(salon_schedule=schedule)
    assert not prefs.open_in_hour(local(2, 8), TZ)
    assert prefs.open_in_hour(local(2, 9), TZ)  # 09:00-10:00 holds 09:30
    assert prefs.open_in_hour(local(2, 18), TZ)  # 18:00-19:00 holds 18:00-18:30
    assert not prefs.open_in_hour(local(2, 19), TZ)


def test_whole_hour_schedules_are_unchanged_by_open_in_hour():
    prefs = RadiatorPrefs()
    assert [prefs.open_in_hour(local(2, h), TZ) for h in (8, 9, 18, 19)] == [
        False,
        True,
        True,
        False,
    ]


@pytest.mark.parametrize(
    "room",
    [
        {"entity": "climate.a", "t_room_c": "abc"},
        {"entity": "climate.a", "t_room_c": 0},
        {"entity": "climate.a", "t_room_c": float("nan")},
        {"entity": "climate.a", "t_room_c": 20.0, "band_c": float("inf")},
    ],
)
def test_invalid_guest_rooms_raise_configuration_error(room):
    with pytest.raises(ConfigurationError):
        RadiatorPrefs.from_dict({"guest_rooms": [room]})


def test_a_non_finite_salon_number_is_invalid():
    with pytest.raises(ConfigurationError):
        RadiatorPrefs(salon_closed_floor_c=float("nan"))


@pytest.mark.parametrize(
    "bad", [{"salon_open_floor_c": 20.5}, {"salon_open_floor_c": 16.5}]
)
def test_the_open_floor_lies_between_the_closed_floor_and_the_target(bad):
    with pytest.raises(ConfigurationError):
        RadiatorPrefs(**bad)
