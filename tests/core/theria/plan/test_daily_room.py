from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.daily_room import (
    MAX_OUTDOOR_C,
    DailyData,
    daily_means,
    fit_daily_room,
    room_params_from_fit,
)

TRUE_GAIN = 0.4
TRUE_RATE = 0.2  # per day: time constant 5 days


def make_data(
    gain=TRUE_GAIN,
    rate=TRUE_RATE,
    seasons=3,
    days_per_season=150,
    seed=1,
    season_gains=None,
):
    """Daily means from a known model. Flow varies independently of the weather, like the
    price-controlled years, and every season has its own unmeasured heat gain `b`."""
    rng = np.random.default_rng(seed)
    start = datetime(2022, 10, 1, tzinfo=UTC)
    days, room, t_out, flow, sun = [], [], [], [], []
    r = 19.5
    for s in range(seasons):
        g = gain if season_gains is None else season_gains[s]
        k_flow, k_out = g * rate, (1 - g) * rate
        b = rng.uniform(-0.4, 0.4)
        for d in range(days_per_season):
            day = start + timedelta(days=s * 365 + d)
            t = 4 + 5 * np.sin(2 * np.pi * d / 60) + rng.normal(0, 1.5)
            f = 37 + rng.choice([-5, -2.5, 0, 2.5, 5]) - 0.5 * (t - 4)
            sun_d = max(0.0, 100 + 80 * np.sin(2 * np.pi * d / 90) + rng.normal(0, 30))
            days.append(day)
            room.append(r)
            t_out.append(t)
            flow.append(f)
            sun.append(sun_d)
            r = r + k_out * (t - r) + k_flow * (f - r) + b
    return DailyData(
        days=days,
        room=np.array(room),
        t_out=np.array(t_out),
        flow=np.array(flow),
        sun=np.array(sun),
    )


def test_fit_recovers_the_response_to_flow_and_the_time_constant():
    fit = fit_daily_room(make_data())
    assert fit.gain_c_per_c == pytest.approx(TRUE_GAIN, abs=0.03)
    assert fit.rate_per_day == pytest.approx(TRUE_RATE, abs=0.02)
    assert fit.time_constant_days == pytest.approx(1 / TRUE_RATE, rel=0.15)
    assert [s.season for s in fit.seasons] == [2022, 2023, 2024]
    assert all(abs(s.gain - TRUE_GAIN) < 0.05 for s in fit.seasons)


def test_the_planner_parameters_are_the_hourly_version_of_the_daily_ones():
    fit = fit_daily_room(make_data())
    params = room_params_from_fit(fit)
    assert params.k_slab == pytest.approx(fit.gain_c_per_c * fit.rate_per_day / 24)
    assert params.k_out == pytest.approx((1 - fit.gain_c_per_c) * fit.rate_per_day / 24)
    assert (
        params.tau_slab_h == 1.0 and params.k_sun == 0.0 and params.drift_c_per_h == 0.0
    )


def test_cross_validation_scores_are_reported_for_one_to_three_days():
    fit = fit_daily_room(make_data())
    assert set(fit.cv_rmse_c) == {1, 2, 3} == set(fit.cv_persistence_rmse_c)
    assert all(np.isfinite(v) for v in fit.cv_rmse_c.values())


def test_a_single_season_cannot_show_that_the_response_is_stable():
    with pytest.raises(ModelFitError, match="seasons"):
        fit_daily_room(make_data(seasons=1))


def test_no_response_to_flow_is_rejected():
    with pytest.raises(ModelFitError, match="response"):
        fit_daily_room(make_data(gain=0.0))


def test_seasons_that_disagree_about_the_response_are_rejected():
    with pytest.raises(ModelFitError, match="differ"):
        fit_daily_room(make_data(season_gains=[0.5, 0.08, 0.5]))


def test_daily_means_need_nearly_a_whole_day_of_data():
    start = datetime(2026, 1, 1, tzinfo=UTC)
    hours = [start + timedelta(hours=i) for i in range(48)]
    values = np.arange(48, dtype=float)
    values[5:15] = np.nan  # 10 missing hours on day one
    out = daily_means(hours, values)
    assert np.isnan(out[0]) and out[1] == pytest.approx(np.mean(np.arange(24, 48)))


def test_days_that_are_too_warm_for_heating_are_left_out_of_the_fit():
    cold = make_data()
    warm_days = 120  # a summer in every season: the heating is off, flow means nothing
    rng = np.random.default_rng(5)
    days, room, t_out, flow, sun = [], [], [], [], []
    r = 22.0
    for s in range(3):
        for d in range(warm_days):
            days.append(datetime(2023 + s, 6, 1, tzinfo=UTC) + timedelta(days=d))
            t = 20 + rng.normal(0, 1)
            f = 25 + rng.normal(0, 3)  # random flow, no effect on the room
            room.append(r)
            t_out.append(t)
            flow.append(f)
            sun.append(200.0)
            r = r + 0.15 * (t - r) + 0.5  # summer sun
    mixed = DailyData(
        days=cold.days + days,
        room=np.concatenate([cold.room, room]),
        t_out=np.concatenate([cold.t_out, t_out]),
        flow=np.concatenate([cold.flow, flow]),
        sun=np.concatenate([cold.sun, sun]),
    )
    assert MAX_OUTDOOR_C == 12.0
    assert fit_daily_room(mixed).gain_c_per_c == pytest.approx(
        fit_daily_room(cold).gain_c_per_c, abs=0.02
    )


def test_seasons_that_disagree_about_how_fast_the_flow_acts_are_rejected():
    # same long-run gain (0.4), but one season reacts ten times slower
    rates = [0.3, 0.3, 0.04]
    parts = [make_data(rate=r, seasons=1, seed=i) for i, r in enumerate(rates)]
    days = []
    for i, part in enumerate(parts):
        days += [d + timedelta(days=365 * i) for d in part.days]
    data = DailyData(
        days=days,
        room=np.concatenate([p.room for p in parts]),
        t_out=np.concatenate([p.t_out for p in parts]),
        flow=np.concatenate([p.flow for p in parts]),
        sun=np.concatenate([p.sun for p in parts]),
    )
    with pytest.raises(ModelFitError, match="differ"):
        fit_daily_room(data)
