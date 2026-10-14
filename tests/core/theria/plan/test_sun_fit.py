from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.sun_fit import MAX_SWING_C_PER_WM2, fit_sun

START = datetime(2026, 1, 5, 0, tzinfo=UTC)


def filtered(sun, lag_h):
    out = np.empty(len(sun))
    value, k = float(sun[0]), 1.0 - np.exp(-1.0 / lag_h)
    for i, s in enumerate(sun):
        out[i] = value
        value += (s - value) * k
    return out


def world(swing, lag_h=2.0, days=60, seed=2, clear=0.7, noise=0.03):
    """Hourly room = a slow level + swing * filtered sun (the sun's swing is quick and does not pile up)."""
    rng = np.random.default_rng(seed)
    n = days * 24
    hours = [START + timedelta(hours=i) for i in range(n)]
    hod = np.array([h.hour for h in hours])
    peak = np.where(rng.random(days) < clear, 450.0, 120.0)[np.arange(n) // 24]
    sun = np.where((hod >= 8) & (hod < 16), peak * np.sin(np.pi * (hod - 8) / 8), 0.0)
    level = 19.0 + 0.4 * np.sin(2 * np.pi * np.arange(n) / (24 * 11))  # slow wandering
    room = level + swing * filtered(sun, lag_h) + rng.normal(0, noise, n)
    t_out = np.full(n, 3.0)
    return hours, room, t_out, sun


def test_a_known_swing_and_lag_are_recovered():
    result = fit_sun(*world(0.01))
    assert result.swing_c_per_wm2 == pytest.approx(0.01, rel=0.2)
    assert result.lag_h == 2.0
    assert result.rmse_after_c < result.rmse_before_c


def test_a_house_the_sun_does_not_warm_gets_no_swing():
    assert fit_sun(*world(0.0)).swing_c_per_wm2 < 0.0005


def test_too_few_sunny_hours_are_refused():
    with pytest.raises(ModelFitError, match="sunny hours"):
        fit_sun(*world(0.01, days=8, clear=0.0))


def test_an_implausibly_large_swing_is_refused():
    with pytest.raises(ModelFitError, match="implausible"):
        fit_sun(*world(0.2))
    assert MAX_SWING_C_PER_WM2 < 0.2
