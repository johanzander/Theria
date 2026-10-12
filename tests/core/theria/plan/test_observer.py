from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.observer import (
    GAIN,
    HALF_LIFE_H,
    MAX_DRIFT_C_PER_H,
    DriftObserver,
)

NOW = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)


def test_a_room_cooling_faster_than_predicted_gives_a_falling_drift():
    observer = DriftObserver()
    observer.update(NOW, measured_c=19.0, predicted_c=19.6, lead_h=3.0)
    assert observer.drift_c_per_h == pytest.approx(GAIN * -0.2)


def test_the_estimate_follows_repeated_errors_and_is_capped():
    observer = DriftObserver()
    for k in range(50):
        observer.update(
            NOW + timedelta(minutes=15 * k),
            measured_c=15.0,
            predicted_c=19.0,
            lead_h=2.0,
        )
    assert observer.drift_c_per_h == pytest.approx(-MAX_DRIFT_C_PER_H, abs=1e-6)


def test_without_new_information_the_estimate_fades():
    observer = DriftObserver()
    observer.update(NOW, measured_c=19.0, predicted_c=19.6, lead_h=3.0)
    before = observer.drift_c_per_h
    assert observer.at(NOW + timedelta(hours=HALF_LIFE_H)) == pytest.approx(before / 2)


def test_the_correction_reaches_less_far_the_further_ahead_the_hour_is():
    observer = DriftObserver()
    last = NOW
    for k in range(20):
        last = NOW + timedelta(minutes=15 * k)
        observer.update(last, measured_c=18.5, predicted_c=19.5, lead_h=2.0)
    starts = [last + timedelta(hours=h) for h in range(12)]
    correction = observer.correction(starts, last)
    assert correction[0] < 0 and abs(correction[0]) > abs(correction[-1]) > 0
    assert np.all(np.diff(np.abs(correction)) <= 1e-12)


def test_a_new_observer_changes_nothing():
    starts = [NOW + timedelta(hours=h) for h in range(4)]
    assert not DriftObserver().correction(starts, NOW).any()


# ---- in the service

from datetime import timedelta as _td  # noqa: E402

from .test_room_level import room_at  # noqa: E402
from .test_service_active import EVENING, make, run_cycle  # noqa: E402


def sensor_reads(ha, value):
    original = ha.get_state

    def reading(entity_id):
        state = original(entity_id)
        if entity_id == "sensor.lk_bibliotek_temperature":
            return {"state": str(value), "attributes": {}}
        return state

    ha.get_state = reading


def test_a_room_that_cools_faster_than_the_plan_expected_is_noticed(settings):
    service, ha, _, _, clock = make(settings)
    run_cycle(service)
    assert (
        service.parameters()["observer_drift_c_per_h"] == 0.0
    )  # nothing to compare yet
    sensor_reads(ha, 19.0)  # the plan from three hours ago expected about 19.8
    clock[0] = EVENING + _td(hours=3)
    run_cycle(service)
    assert service.parameters()["observer_drift_c_per_h"] < 0


def test_a_plan_that_is_too_recent_to_judge_is_not_compared(settings):
    service, ha, _, _, clock = make(settings)
    run_cycle(service)
    sensor_reads(ha, 17.0)
    clock[0] = EVENING + _td(minutes=30)
    run_cycle(service)
    assert service.parameters()["observer_drift_c_per_h"] == 0.0


def test_a_sunny_peak_is_not_an_error_because_the_level_is_compared(settings):
    service, ha, _, _, clock = make(settings, actuals_loader=room_at(19.8))
    run_cycle(service)
    sensor_reads(ha, 22.5)  # the sun: the level stays at the day's median
    clock[0] = EVENING + _td(hours=3)
    run_cycle(service)
    assert abs(service.parameters()["observer_drift_c_per_h"]) < 0.02
