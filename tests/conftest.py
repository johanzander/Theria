"""
Pytest configuration and fixtures for Theria tests.

Follows Home Assistant testing best practices with reusable fixtures,
parametrized tests, and clear test organization.
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.theria.entity_thermal_learner import (
    ClimateEntityThermalLearner,
    EntityMeasurement,
)


@pytest.fixture
def base_time():
    """Base timestamp for all tests (timezone-aware UTC)."""
    return datetime(2025, 12, 29, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def entity_id():
    """Default entity ID for tests."""
    return "climate.test_radiator"


@pytest.fixture
def learner(entity_id):
    """Create a fresh ClimateEntityThermalLearner instance."""
    return ClimateEntityThermalLearner(
        entity_id=entity_id, max_history_hours=168, min_samples_for_confidence=100
    )


@pytest.fixture
def create_measurement(base_time, entity_id):
    """Factory fixture to create EntityMeasurement objects."""

    def _create(
        offset_minutes: int = 0,
        temperature: float = 20.0,
        outdoor_temp: float = 5.0,
        heating_active: bool = False,
        target_temp: float | None = 21.0,
    ) -> EntityMeasurement:
        return EntityMeasurement(
            timestamp=base_time + timedelta(minutes=offset_minutes),
            temperature=temperature,
            outdoor_temp=outdoor_temp,
            heating_active=heating_active,
            target_temp=target_temp,
        )

    return _create


@pytest.fixture
def sample_heating_period(create_measurement):
    """
    Sample heating period: 2 hours, temperature rising from 18°C to 20°C.
    Expected rate: +1.0°C/h
    """
    measurements = [
        create_measurement(0, temperature=18.0, heating_active=False, target_temp=22.0),
        create_measurement(15, temperature=18.0, heating_active=True, target_temp=22.0),
        create_measurement(
            30, temperature=18.25, heating_active=True, target_temp=22.0
        ),
        create_measurement(45, temperature=18.5, heating_active=True, target_temp=22.0),
        create_measurement(
            60, temperature=18.75, heating_active=True, target_temp=22.0
        ),
        create_measurement(75, temperature=19.0, heating_active=True, target_temp=22.0),
        create_measurement(
            90, temperature=19.25, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            105, temperature=19.5, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            120, temperature=19.75, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            135, temperature=20.0, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            150, temperature=20.0, heating_active=False, target_temp=22.0
        ),
    ]
    return measurements


@pytest.fixture
def sample_cooling_period(create_measurement):
    """
    Sample cooling period: 4 hours, temperature falling from 22°C to 20°C.
    Expected rate: -0.5°C/h
    """
    measurements = [
        create_measurement(0, temperature=22.0, heating_active=True, target_temp=21.0),
        create_measurement(
            15, temperature=22.0, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            30, temperature=21.9, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            60, temperature=21.75, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            90, temperature=21.5, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            120, temperature=21.25, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            150, temperature=21.0, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            180, temperature=20.75, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            210, temperature=20.5, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            240, temperature=20.25, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            255, temperature=20.0, heating_active=False, target_temp=21.0
        ),
        create_measurement(
            270, temperature=20.0, heating_active=True, target_temp=21.0
        ),
    ]
    return measurements


@pytest.fixture
def sample_mixed_periods(create_measurement):
    """
    Mixed heating and cooling periods for duration-weighted averaging tests.

    Period 1 (Heating): 1 hour, 18°C → 20°C = +2.0°C/h
    Period 2 (Cooling): 3 hours, 20°C → 19.5°C = -0.17°C/h
    Period 3 (Heating): 2 hours, 19.5°C → 21.0°C = +0.75°C/h

    Expected weighted heating rate: (2.0*1 + 0.75*2) / (1+2) = 3.5/3 = +1.17°C/h
    Expected weighted cooling rate: -0.5 / 3 = -0.17°C/h
    """
    measurements = [
        # Period 1: Heating (1 hour, +2°C/h)
        create_measurement(0, temperature=18.0, heating_active=False, target_temp=22.0),
        create_measurement(15, temperature=18.0, heating_active=True, target_temp=22.0),
        create_measurement(30, temperature=18.5, heating_active=True, target_temp=22.0),
        create_measurement(45, temperature=19.0, heating_active=True, target_temp=22.0),
        create_measurement(60, temperature=19.5, heating_active=True, target_temp=22.0),
        create_measurement(75, temperature=20.0, heating_active=True, target_temp=22.0),
        create_measurement(
            90, temperature=20.0, heating_active=False, target_temp=22.0
        ),
        # Period 2: Cooling (3 hours, -0.17°C/h)
        create_measurement(
            105, temperature=20.0, heating_active=False, target_temp=20.0
        ),
        create_measurement(
            150, temperature=19.9, heating_active=False, target_temp=20.0
        ),
        create_measurement(
            210, temperature=19.75, heating_active=False, target_temp=20.0
        ),
        create_measurement(
            270, temperature=19.5, heating_active=False, target_temp=20.0
        ),
        create_measurement(
            285, temperature=19.5, heating_active=True, target_temp=22.0
        ),
        # Period 3: Heating (2 hours, +0.75°C/h)
        create_measurement(
            300, temperature=19.5, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            345, temperature=20.0, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            390, temperature=20.5, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            405, temperature=21.0, heating_active=True, target_temp=22.0
        ),
        create_measurement(
            420, temperature=21.0, heating_active=False, target_temp=22.0
        ),
    ]
    return measurements


@pytest.fixture
def zero_change_period(create_measurement):
    """
    Period with no temperature change (should be filtered out).
    Heating is ON but temperature remains constant at 21°C.
    """
    measurements = [
        create_measurement(0, temperature=21.0, heating_active=False, target_temp=21.0),
        create_measurement(15, temperature=21.0, heating_active=True, target_temp=21.0),
        create_measurement(30, temperature=21.0, heating_active=True, target_temp=21.0),
        create_measurement(45, temperature=21.0, heating_active=True, target_temp=21.0),
        create_measurement(60, temperature=21.0, heating_active=True, target_temp=21.0),
        create_measurement(
            75, temperature=21.0, heating_active=False, target_temp=21.0
        ),
    ]
    return measurements


@pytest.fixture
def short_duration_period(create_measurement):
    """
    Period shorter than minimum duration (15 minutes).
    Should be filtered out.
    """
    measurements = [
        create_measurement(0, temperature=20.0, heating_active=False, target_temp=22.0),
        create_measurement(5, temperature=20.0, heating_active=True, target_temp=22.0),
        create_measurement(10, temperature=20.1, heating_active=True, target_temp=22.0),
        create_measurement(
            15, temperature=20.1, heating_active=False, target_temp=22.0
        ),
    ]
    return measurements


@pytest.fixture
def maintenance_mode_period(create_measurement):
    """
    Heating period in maintenance mode (temp close to target).
    Should be classified as "maintenance" not "heating".
    """
    measurements = [
        create_measurement(0, temperature=21.5, heating_active=False, target_temp=22.0),
        create_measurement(15, temperature=21.5, heating_active=True, target_temp=22.0),
        create_measurement(30, temperature=21.6, heating_active=True, target_temp=22.0),
        create_measurement(45, temperature=21.7, heating_active=True, target_temp=22.0),
        create_measurement(60, temperature=21.8, heating_active=True, target_temp=22.0),
        create_measurement(
            75, temperature=21.8, heating_active=False, target_temp=22.0
        ),
    ]
    return measurements
