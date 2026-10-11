from datetime import UTC, datetime

import pytest

from core.theria.exceptions import WeatherDataError
from core.theria.plan.weather import parse_open_meteo


def payload(**overrides):
    base = {
        "hourly": {
            "time": ["2026-10-12T13:00", "2026-10-12T14:00"],
            "shortwave_radiation": [372.0, 329.0],
            "cloud_cover": [52, 87],
            "temperature_2m": [12.7, 11.8],
        }
    }
    base["hourly"].update(overrides)
    return base


def test_label_is_end_of_hour_so_start_is_one_hour_earlier():
    hours = parse_open_meteo(payload())
    assert hours[0].start == datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert hours[0].shortwave_wm2 == 372.0
    assert hours[1].temperature_c == 11.8


def test_missing_value_raises():
    with pytest.raises(WeatherDataError):
        parse_open_meteo(payload(shortwave_radiation=[372.0, None]))


def test_missing_field_raises():
    with pytest.raises(WeatherDataError):
        parse_open_meteo({"hourly": {"time": []}})


def test_mismatched_lengths_raise():
    with pytest.raises(WeatherDataError):
        parse_open_meteo(payload(cloud_cover=[52]))
