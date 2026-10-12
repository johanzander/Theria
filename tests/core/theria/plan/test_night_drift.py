from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan import night_drift
from core.theria.plan.night_drift import (
    NightDrift,
    drift_per_hour,
    fit_night_drift,
    night_change_c,
)

START = datetime(2025, 1, 1, tzinfo=UTC)
A, B = (
    -0.5,
    0.06,
)  # true night change: -0.5 C at 0 C outdoors, 0.06 C less cooling per C warmer


def world(
    days=400,
    start_hour=0,
    end_hour=6,
    slope=B,
    late_slope=None,
    late_from=200,
    polluted=(),
    level_slope=0.0,
    swing=0.0,
):
    """Hourly room/outdoor/setpoint series in UTC with a known night cooling."""
    n = days * 24
    hours = [START + timedelta(hours=i) for i in range(n)]
    day = np.arange(n) // 24
    t = 4 + 9 * np.sin(2 * np.pi * day / 365 - 1.5) + 2 * np.sin(day * 1.7)
    room = np.full(n, 19.5)
    rts = np.full(n, 18.0)
    length = (end_hour - start_hour) % 24
    for d in range(days):
        s = d * 24 + start_hour
        if s + length >= n:
            break
        tt = float(np.mean(t[s : s + length]))
        use = late_slope if late_slope is not None and d >= late_from else slope
        level = 19.5 + 0.8 * np.sin(
            d * 0.9
        )  # the room starts the night warmer or cooler
        change = A + use * tt + level_slope * (level - 19.5)
        if d in polluted:
            rts[s - 3 : s + length] += 3.0  # moved far: not a clean night
            change = -2.0
        if swing and d % 2:
            # the plan lowered the setpoint in the second half of the night: the room is a little
            # cooler than the night alone would make it
            rts[s + length // 2 : s + length + 1] -= swing
            change -= (
                night_drift.SETPOINT_EFFECT_C_PER_C_H * swing * (length - length // 2)
            )
        room[s : s + length + 1] = level + change * np.arange(length + 1) / length
    return hours, room, t, rts


def fit(data, start_hour=0, end_hour=6):
    hours, room, t, rts = data
    return fit_night_drift(hours, room, t, rts, start_hour, end_hour, "UTC", 19.5)


def test_the_night_cooling_is_recovered_as_a_function_of_the_outdoor_temperature():
    drift = fit(world())
    assert drift.intercept_c == pytest.approx(A, abs=0.01)
    assert drift.slope_c_per_c == pytest.approx(B, abs=0.002)
    assert drift.night_hours == 6 and drift.nights > 300
    assert night_change_c(drift, -7.0, 19.5) == pytest.approx(A + B * -7.0, abs=0.03)
    assert drift_per_hour(drift, -7.0, 19.5) == pytest.approx(
        (A + B * -7.0) / 6, abs=0.01
    )


def test_a_night_across_midnight_is_handled():
    drift = fit(world(start_hour=22, end_hour=5), 22, 5)
    assert drift.night_hours == 7
    assert drift.slope_c_per_c == pytest.approx(B, abs=0.002)


def test_nights_where_the_setpoint_was_moved_do_not_count():
    clean = fit(world())
    messy = fit(world(polluted=range(0, 400, 7)))
    assert messy.slope_c_per_c == pytest.approx(clean.slope_c_per_c, abs=0.002)
    assert messy.nights < clean.nights


def test_recent_nights_count_more_than_old_ones(monkeypatch):
    data = world(
        slope=0.02, late_slope=0.06, late_from=320
    )  # the house changed 80 days ago
    weighted = fit(data).slope_c_per_c
    monkeypatch.setattr(
        night_drift, "HALF_LIFE_DAYS", 1e9
    )  # every night counts the same
    unweighted = fit(data).slope_c_per_c
    assert weighted > unweighted + 0.002
    assert abs(weighted - 0.06) < abs(unweighted - 0.06)


def test_the_last_nights_are_listed_with_the_predicted_and_the_measured_change():
    drift = fit(world())
    assert len(drift.recent) == 14
    last = drift.recent[-1]
    assert last.measured_change_c == pytest.approx(last.predicted_change_c, abs=0.03)
    assert drift.recent[0].day < last.day


@pytest.mark.parametrize(
    "build",
    [
        lambda: world(days=20),  # too few nights
        lambda: world(slope=-0.05),  # colder nights that cool LESS is not physical
    ],
)
def test_unconvincing_fits_are_refused(build):
    with pytest.raises(ModelFitError):
        fit(build())


def test_no_temperature_spread_is_refused():
    hours, room, t, rts = world()
    with pytest.raises(ModelFitError):
        fit_night_drift(hours, room, np.full_like(t, 5.0), rts, 0, 6, "UTC", 19.5)


def test_an_empty_night_window_is_refused():
    with pytest.raises(ModelFitError):
        fit(world(), 3, 3)


def test_night_drift_is_a_plain_value_object():
    drift = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 100, 0.1, ())
    assert drift.nights == 100


def test_the_pattern_cools_the_night_and_warms_the_day_by_the_same_amount():
    from core.theria.plan.night_drift import drift_pattern

    drift = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 100, 0.1, ())
    starts = [START + timedelta(hours=i) for i in range(36)]  # starts at 00:00 UTC
    t_out = np.full(36, -4.0)
    pattern = drift_pattern(drift, lambda moment: moment.hour < 6, starts, t_out, 19.5)
    assert pattern[:6] == pytest.approx([(A + B * -4.0) / 6] * 6)
    assert np.sum(pattern) == pytest.approx(0.0, abs=1e-9)
    assert pattern[6] > 0 and pattern[6] == pytest.approx(pattern[10])


def test_without_a_night_in_the_horizon_there_is_no_pattern():
    from core.theria.plan.night_drift import drift_pattern

    drift = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 100, 0.1, ())
    starts = [START + timedelta(hours=8 + i) for i in range(10)]
    assert not np.any(
        drift_pattern(drift, lambda m: m.hour < 6, starts, np.full(10, 0.0), 19.5)
    )


def test_a_warmer_room_cools_more_over_the_night():
    drift = fit(world(level_slope=-0.2))
    assert drift.level_c_per_c == pytest.approx(-0.2, abs=0.02)
    assert drift.slope_c_per_c == pytest.approx(B, abs=0.003)
    warm, cool = night_change_c(drift, 5.0, 20.5), night_change_c(drift, 5.0, 18.5)
    assert warm < cool  # the warmer room loses more
    assert warm - cool == pytest.approx(-0.2 * 2.0, abs=0.05)


def test_a_level_effect_with_the_wrong_sign_is_left_out_not_trusted():
    drift = fit(
        world(level_slope=+0.3)
    )  # a warmer room that cools LESS is not physical
    assert drift.level_c_per_c == 0.0
    assert drift.slope_c_per_c > 0


def test_nights_with_a_setpoint_the_plan_moved_teach_the_model_after_the_setpoint_is_accounted_for():
    plain = fit(world())
    active = fit(world(swing=1.5))
    assert active.intercept_c == pytest.approx(plain.intercept_c, abs=0.02)
    assert active.slope_c_per_c == pytest.approx(plain.slope_c_per_c, abs=0.003)
    assert active.nights > plain.nights * 0.95
