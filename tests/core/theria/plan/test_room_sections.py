from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.room_sections import (
    load_section_measurements,
    room_sections_from_options,
    section_entities,
    silent_entities,
)

START = datetime(2026, 11, 10, 0, tzinfo=UTC)

OPTIONS = {
    "room_sections": [
        {
            "id": "guest",
            "name": "Gårdshus",
            "rooms": [{"name": "Kök", "entity": "sensor.kitchen_temperature"}],
            "thermostats": [
                {
                    "name": "Kök",
                    "entity": "climate.kitchen",
                    "demand_entity": "sensor.kitchen_heating",
                }
            ],
        },
        {"id": "upper", "name": "Ovanvåning", "rooms": []},
    ]
}


class Source:
    """Points at minutes after START per (entity, field)."""

    def __init__(self, data):
        self.data = data

    def series(self, entity, start, stop, field=None):
        return [
            (START + timedelta(minutes=m), float(v))
            for m, v in self.data.get((entity, field), [])
            if start <= START + timedelta(minutes=m) < stop
        ]


def test_sections_are_read_from_the_options():
    sections = room_sections_from_options(OPTIONS)
    assert [s.id for s in sections] == ["guest", "upper"]
    assert sections[0].thermostats[0].demand_field is None
    assert sections[1].thermostats == ()
    assert room_sections_from_options({}) == ()


def test_a_repeated_section_id_is_refused():
    with pytest.raises(ConfigurationError):
        room_sections_from_options({"room_sections": OPTIONS["room_sections"] * 2})


def test_rooms_setpoints_and_demand_per_hour():
    sections = room_sections_from_options(OPTIONS)
    source = Source(
        {
            ("sensor.kitchen_temperature", None): [(10, 19.0), (40, 21.0), (70, 20.5)],
            # set the evening before the day: still the setpoint all day
            ("climate.kitchen", "temperature"): [(-600, 20.0)],
            ("sensor.kitchen_heating", None): [(5, 0), (35, 40)],
        }
    )
    measured = load_section_measurements(
        source, sections, START, START + timedelta(hours=3), 3
    )
    guest = measured["guest"]
    assert guest.rooms["Kök"][:2] == pytest.approx([20.0, 20.5])
    assert guest.rooms["Kök"][2] == pytest.approx(
        20.5
    )  # held while the sensor is silent
    assert guest.setpoints["Kök"] == pytest.approx([20.0, 20.0, 20.0])
    assert guest.demand["Kök"][0] == pytest.approx(20.0)
    assert measured["upper"].rooms == {}


def test_a_sensor_home_assistant_has_lost_ends_with_its_last_report():
    sections = room_sections_from_options(OPTIONS)
    states = [
        # last_changed is when the integration was reloaded, not when the battery died
        {"entity_id": "sensor.kitchen_temperature", "state": "unknown"},
        {"entity_id": "climate.kitchen", "state": "unavailable"},
        {"entity_id": "sensor.kitchen_heating", "state": "12"},
        {"entity_id": "sensor.other", "state": "unavailable"},
    ]
    silent = silent_entities(states, section_entities(sections))
    assert silent == {"sensor.kitchen_temperature", "climate.kitchen"}
    source = Source(
        {
            ("sensor.kitchen_temperature", None): [(10, 19.0), (65, 21.0)],
            ("climate.kitchen", "temperature"): [(-600, 20.0)],
        }
    )
    measured = load_section_measurements(
        source, sections, START, START + timedelta(hours=4), 4, silent
    )
    guest = measured["guest"]
    # the hour of the last report keeps it; the hours after are unknown, not held
    assert guest.rooms["Kök"][:2] == pytest.approx([19.0, 21.0])
    assert np.isnan(guest.rooms["Kök"][2:]).all()
    assert np.isnan(guest.setpoints["Kök"]).all()  # last set the evening before
    assert guest.offline == {"Kök": START + timedelta(minutes=65)}
    assert measured["upper"].offline == {}
