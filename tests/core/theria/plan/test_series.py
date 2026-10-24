from datetime import UTC, datetime, timedelta, timezone

import numpy as np

from core.theria.plan.series import (
    clean_counter,
    despike,
    fill_flat_gaps,
    hourly_counter_energy,
    hourly_energy_from_counter,
    interpolate_gaps,
    step_fraction,
    to_hourly,
)

START = datetime(2026, 1, 10, 0, tzinfo=UTC)


def at(minutes: int, value: float):
    return (START + timedelta(minutes=minutes), value)


def test_mean_and_last_within_the_hour_and_nan_for_empty_hours():
    points = [at(5, 1.0), at(35, 3.0), at(125, 10.0)]
    mean = to_hourly(points, START, 4, how="mean")
    last = to_hourly(points, START, 4, how="last")
    assert mean[0] == 2.0 and last[0] == 3.0
    assert np.isnan(mean[1]) and mean[2] == 10.0 and np.isnan(mean[3])


def test_local_timestamps_land_in_the_right_utc_hour():
    local = timezone(timedelta(hours=2))
    point = (datetime(2026, 1, 10, 3, 30, tzinfo=local), 7.0)  # 01:30 UTC
    assert to_hourly([point], START, 3)[1] == 7.0


def test_forward_fill_carries_values_only_for_the_allowed_hours():
    out = to_hourly([at(10, 5.0)], START, 6, how="last", ffill_hours=2)
    assert list(out[:3]) == [5.0, 5.0, 5.0]
    assert np.isnan(out[3])


def test_non_finite_points_are_ignored():
    assert np.isnan(to_hourly([at(1, float("nan"))], START, 1)[0])


def test_interpolation_fills_short_gaps_only():
    values = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])
    out = interpolate_gaps(values, max_gap_hours=2)
    assert out[1] == 2.0
    assert np.isnan(out[3:6]).all()


def test_despike_removes_isolated_garbage_and_keeps_real_steps():
    values = np.array([20.0, 20.1, 240.0, 20.2, 20.3, 20.4, 20.5])
    out = despike(values, window=5, max_dev=3.0)
    assert np.isnan(out[2])
    assert np.isfinite(np.delete(out, 2)).all()


def test_counter_differences_drop_resets_and_spikes():
    counter = np.array([100.0, 101.0, 103.0, 5.0, 6.0, 900.0, 7.0, 8.5])
    out = hourly_energy_from_counter(counter, max_kwh_per_hour=4.0)
    assert out[1] == 1.0 and out[2] == 2.0
    assert np.isnan(out[3])  # reset (negative difference)
    assert out[4] == 1.0
    assert np.isnan(out[5]) and np.isnan(out[6])  # spike up and back down
    assert out[7] == 1.5


def test_step_fraction_is_the_share_of_each_hour_the_state_was_on():
    # on 10:05-10:40 (35 min), on again from 12:30 until the next event at 14:15
    events = [at(-120, 0.0), at(5, 1.0), at(40, 0.0), at(150, 1.0), at(255, 0.0)]
    out = step_fraction(events, START, 6)
    assert np.allclose(out[0], 35 / 60)
    assert out[1] == 0.0
    assert np.isclose(out[2], 0.5)
    assert out[3] == 1.0
    assert np.isclose(out[4], 15 / 60)
    assert out[5] == 0.0


def test_step_fraction_is_unknown_before_the_first_event_and_after_the_carry_limit():
    out = step_fraction([at(90, 1.0)], START, 8, max_carry_hours=3)
    assert np.isnan(out[0]) and np.isnan(
        out[1]
    )  # unknown before the first event (01:30)
    assert out[2] == 1.0 and out[4] == 1.0
    assert np.isnan(out[5])  # carried more than 3 h without a new event


def test_clean_counter_keeps_the_real_track_even_when_most_readings_are_garbage():
    # a counter that really reads 938 and often answers 0-16 (the H60 gateway): more garbage than real readings
    points = [at(m, v) for m, v in [(0, 938.0), (5, 3.0), (10, 16.0), (15, 0.0), (20, 938.0), (25, 7.0), (30, 12.0), (35, 938.0), (40, 938.0)]]
    assert [v for _, v in clean_counter(points, 9.0)] == [938.0] * 4


def test_clean_counter_drops_a_spike_up_and_keeps_the_readings_after_it():
    points = [at(m, v) for m, v in [(0, 100.0), (10, 101.0), (20, 9000.0), (30, 102.0), (40, 103.0)]]
    assert [v for _, v in clean_counter(points, 5.0)] == [100.0, 101.0, 102.0, 103.0]


def test_clean_counter_works_on_each_two_weeks_so_a_reset_cannot_erase_the_newer_readings():
    old = [(START + timedelta(days=d), 70000.0 + d) for d in range(3)]
    new = [(START + timedelta(days=20 + d), 700.0 + d) for d in range(3)]
    assert [v for _, v in clean_counter(old + new, 5.0)] == [70000.0, 70001.0, 70002.0, 700.0, 701.0, 702.0]


def test_a_gap_between_equal_readings_is_filled_but_a_gap_between_different_readings_is_not():
    out = fill_flat_gaps(np.array([5.0, np.nan, np.nan, 5.0, np.nan, 7.0, np.nan]))
    assert out.tolist()[:4] == [5.0, 5.0, 5.0, 5.0]
    assert np.isnan(out[4]) and np.isnan(out[6])


def test_a_quiet_counter_with_garbage_readings_is_zero_consumption_not_missing_data():
    # 938 all day with garbage zeros now and then: the heater did not run
    points = [at(m, v) for m, v in [(0, 938.0), (70, 4.0), (130, 938.0), (400, 2.0), (500, 938.0)]]
    out = hourly_counter_energy(points, START, 9, max_kwh_per_hour=9.0, hold_hours=3)
    assert np.isfinite(out[1:9]).all() and (out[1:9] == 0.0).all()
