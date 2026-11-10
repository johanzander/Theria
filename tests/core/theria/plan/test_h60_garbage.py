"""The H60 gateway answers with garbage now and then: none of it may reach the plan."""

from datetime import timedelta

import pytest

from core.theria.exceptions import SensorError
from core.theria.plan.calibration_data import OUTDOOR_PLAUSIBLE_C
from core.theria.plan.forecast_check import load_measured

from .test_forecast_check import HOUR, FakeSource
from .test_service import FakeHA, make_service


def service_reading(settings, entity, value):
    service = make_service(settings)
    original = FakeHA.get_state

    def get_state(entity_id):
        if entity_id == entity:
            return {"state": value, "attributes": {}}
        return original(service._ha, entity_id)

    service._ha.get_state = get_state
    return service


def test_a_garbage_outdoor_reading_stops_the_plan_instead_of_shifting_the_weather(
    settings,
):
    with pytest.raises(SensorError):
        service_reading(settings, settings.outdoor_entity, "59.2").run_once()


def test_a_flow_target_of_zero_stops_the_plan(settings):
    with pytest.raises(SensorError):
        service_reading(settings, settings.flow_setpoint_entity, "0.0").run_once()


def test_a_plausible_outdoor_reading_is_used(settings):
    assert service_reading(settings, settings.outdoor_entity, "-12.0").run_once()


def test_the_checked_flow_target_leaves_out_zero(settings):
    source = FakeSource(
        {
            (settings.flow_setpoint_entity, None): [
                (HOUR + timedelta(minutes=10), 35.0),
                (HOUR + timedelta(minutes=50), 0.0),
            ]
        }
    )
    measured = load_measured(
        source, settings, lambda a, b: [], HOUR, HOUR + timedelta(hours=1)
    )
    assert measured.at(HOUR)["flow_c"] == 35.0


def test_the_checked_outdoor_leaves_out_garbage(settings):
    source = FakeSource(
        {
            (settings.outdoor_entity, None): [
                (HOUR + timedelta(minutes=10), 6.0),
                (HOUR + timedelta(minutes=20), 59.2),
                (HOUR + timedelta(minutes=50), 6.0),
            ]
        }
    )
    measured = load_measured(
        source, settings, lambda a, b: [], HOUR, HOUR + timedelta(hours=1)
    )
    assert measured.at(HOUR)["t_out_c"] == 6.0


def test_the_model_fit_asks_for_plausible_outdoor_hours(settings):
    from .test_calibration import END
    from .world import FakeSource as WorldSource
    from .world import FakeWeatherHistory, World

    asked = {}

    class Source(WorldSource):
        def hourly_means(self, entity_id, start, stop, plausible=None):
            asked[entity_id] = plausible
            return super().hourly_means(entity_id, start, stop)

    from core.theria.plan.calibration_data import load_calibration_inputs

    world = World(days=20)
    load_calibration_inputs(
        Source(world), FakeWeatherHistory(world), settings, END, days=20
    )
    assert asked[settings.outdoor_entity] == OUTDOOR_PLAUSIBLE_C


def test_the_flow_targets_set_with_the_wrong_minimum_outdoor_are_left_out():
    from datetime import UTC, datetime

    import numpy as np

    from core.theria.plan.calibration_data import without_curve_change
    from core.theria.plan.series import hourly_grid

    hours = hourly_grid(datetime(2026, 11, 8, 10, tzinfo=UTC), 32)
    kept = without_curve_change(hours, np.full(32, 40.0))
    assert kept[0] == 40.0  # 10:00 UTC
    # 11:00 UTC, the hour it was set at 11:41, to 16:00 the next day, set back at 16:25
    assert np.isnan(kept[1]) and np.isnan(kept[30])
    assert kept[31] == 40.0  # 17:00 UTC
