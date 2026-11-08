from datetime import UTC, datetime, timedelta

import pytest

from core.theria.plan.disturbance import (
    GAIN,
    HALF_LIFE_H,
    MAX_DRIFT_C_PER_H,
    DisturbanceEstimator,
    Sample,
)
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS, step

NOW = datetime(2026, 10, 12, 2, 0, tzinfo=UTC)
LIMIT = 22.0
PARAMS = PRIOR_ROOM_PARAMS


def simulate(
    estimator, extra_c_per_h=0.0, minutes=15, steps=40, valves=None, flow=None
):
    """Feed samples from a room that follows the model plus `extra_c_per_h`."""
    room, slab, sun_f = (
        19.0,
        23.5,
        0.0,
    )  # the estimator starts the slab half way to the flow
    dt = minutes / 60
    estimator.update(Sample(NOW, room, 28.0, 1.0, 5.0, 0.0), PARAMS, LIMIT)
    for k in range(1, steps + 1):
        valve = 1.0 if valves is None else valves(k)
        f = 28.0 if flow is None else flow(k)
        room, slab, sun_f = step(
            PARAMS,
            room,
            slab,
            sun_f,
            f,
            5.0,
            0.0,
            room + 10.0,
            dt_h=dt,
            valve_scale=valve,
        )
        room += extra_c_per_h * dt
        estimator.update(
            Sample(
                NOW + timedelta(minutes=minutes * k), float(room), f, valve, 5.0, 0.0
            ),
            PARAMS,
            LIMIT,
        )
    return NOW + timedelta(minutes=minutes * steps)


def test_a_room_warming_faster_than_the_model_gives_a_positive_drift():
    estimator = DisturbanceEstimator()
    simulate(estimator, extra_c_per_h=0.1)
    assert 0.05 < estimator.drift_c_per_h <= MAX_DRIFT_C_PER_H


def test_the_plans_own_effect_is_not_read_as_drift():
    estimator = DisturbanceEstimator()
    # valves opening and closing and the flow changing: the model knows, the drift stays ~0
    simulate(
        estimator,
        valves=lambda k: 1.0 if (k // 8) % 2 == 0 else 0.0,
        flow=lambda k: 24.0 + (k % 16),
    )
    assert abs(estimator.drift_c_per_h) < 0.01


def test_the_estimate_is_capped():
    estimator = DisturbanceEstimator()
    simulate(estimator, extra_c_per_h=2.0, steps=80)
    assert estimator.drift_c_per_h == pytest.approx(MAX_DRIFT_C_PER_H, abs=1e-6)


def test_without_new_information_the_estimate_fades():
    estimator = DisturbanceEstimator()
    last = simulate(estimator, extra_c_per_h=0.1)
    before = estimator.at(last)
    assert estimator.at(last + timedelta(hours=HALF_LIFE_H)) == pytest.approx(
        before / 2
    )


def test_a_long_gap_between_samples_is_not_judged():
    estimator = DisturbanceEstimator()
    estimator.update(Sample(NOW, 19.0, 28.0, 1.0, 5.0, 0.0), PARAMS, LIMIT)
    estimator.update(
        Sample(NOW + timedelta(hours=5), 25.0, 28.0, 1.0, 5.0, 0.0), PARAMS, LIMIT
    )
    assert estimator.drift_c_per_h == 0.0


def test_while_the_sun_is_not_modelled_sunny_samples_do_not_teach_it():
    estimator = DisturbanceEstimator()
    estimator.update(Sample(NOW, 19.0, 28.0, 1.0, 5.0, 600.0), PARAMS, LIMIT)
    estimator.update(
        Sample(NOW + timedelta(minutes=15), 19.4, 28.0, 1.0, 5.0, 600.0), PARAMS, LIMIT
    )
    assert estimator.drift_c_per_h == 0.0


def test_a_sensor_that_reads_in_tenths_does_not_look_like_a_drift():
    """0.1 C of rounding over a quarter of an hour would be 0.4 C/h of imagined drift; the estimate
    is judged over at least an hour, so it stays within the rounding."""
    estimator = DisturbanceEstimator()
    room, slab, sun_f = 19.0, 23.5, 0.0
    estimator.update(Sample(NOW, room, 28.0, 1.0, 5.0, 0.0), PARAMS, LIMIT)
    worst = 0.0
    for k in range(1, 49):
        room, slab, sun_f = step(
            PARAMS,
            room,
            slab,
            sun_f,
            28.0,
            5.0,
            0.0,
            room + 10.0,
            dt_h=0.25,
            valve_scale=1.0,
        )
        read = round(float(room), 1)
        estimator.update(
            Sample(NOW + timedelta(minutes=15 * k), read, 28.0, 1.0, 5.0, 0.0),
            PARAMS,
            LIMIT,
        )
        worst = max(worst, abs(estimator.drift_c_per_h))
    assert worst <= 0.06


def test_the_gain_is_a_share_of_the_new_error():
    estimator = DisturbanceEstimator()
    estimator.update(Sample(NOW, 19.0, 28.0, 1.0, 5.0, 0.0), PARAMS, LIMIT)
    predicted, _, _ = step(
        PARAMS, 19.0, 23.5, 0.0, 28.0, 5.0, 0.0, 29.0, dt_h=1.0, valve_scale=1.0
    )
    estimator.update(
        Sample(NOW + timedelta(hours=1), 19.05, 28.0, 1.0, 5.0, 0.0), PARAMS, LIMIT
    )
    assert estimator.drift_c_per_h == pytest.approx(GAIN * (19.05 - predicted))


# ---- in the service

from datetime import timedelta as _td  # noqa: E402

from .test_service_active import EVENING, make, run_cycle  # noqa: E402


def sensor_reads(ha, value):
    original = ha.get_state

    def reading(entity_id):
        state = original(entity_id)
        if entity_id == "sensor.lk_bibliotek_temperature":
            return {"state": str(value), "attributes": {}}
        return state

    ha.get_state = reading


def test_a_room_that_cools_faster_than_the_model_expected_is_noticed(settings):
    service, ha, _, _, clock = make(settings)  # 16:20 UTC: dark
    run_cycle(service)
    assert service.parameters()["disturbance_c_per_h"] == 0.0  # nothing to compare yet
    sensor_reads(ha, 19.0)  # the model expected the room to stay near 19.8
    clock[0] = EVENING + _td(hours=1)
    run_cycle(service)
    assert service.parameters()["disturbance_c_per_h"] < 0


def test_a_sunny_rise_is_not_taken_as_a_disturbance_while_the_sun_is_not_modelled(
    settings,
):
    noon = EVENING.replace(hour=12, minute=0)
    service, ha, _, _, clock = make(settings, clock=lambda: noon)
    run_cycle(service)
    sensor_reads(ha, 22.5)
    clock_after = noon + _td(minutes=15)
    service._now = lambda: clock_after
    run_cycle(service)
    assert service.parameters()["disturbance_c_per_h"] == 0.0


def test_with_the_suns_swing_known_the_plan_starts_from_the_room_minus_the_swing(
    settings,
):
    from dataclasses import replace

    afternoon = EVENING.replace(hour=13, minute=0)  # the fake sun shines 09-15 UTC
    service, ha, *_ = make(settings, clock=lambda: afternoon)
    service._room = replace(service._room, sun_swing_c_per_wm2=0.006)
    run_cycle(service)
    initial = service.latest().initial
    assert initial.room_c < 19.8 - 0.3  # the sensor reads 19.8, part of it is the sun
    assert initial.sun_filtered_wm2 > 100
