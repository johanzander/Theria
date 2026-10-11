from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.warmup import fit_warmup

from .test_calibration import calibrator
from .world import END, World

GAIN, RATE = 0.4, 0.2  # what TRUE_ROOM is built from


START = datetime(2024, 1, 1, tzinfo=UTC)


def linear_room(days=400, gain=GAIN, rate=RATE, seed=5, noise=0.03):
    """A room that follows the daily response model exactly (no valves): x' = rate (gain F - x)."""
    rng = np.random.default_rng(seed)
    n = days * 24
    hours = [START + timedelta(hours=i) for i in range(n)]
    rts = np.repeat(rng.uniform(16.5, 20.5, n // 60 + 1), 60)[:n].round(1)
    t_out = (
        4 + 8 * np.sin(2 * np.pi * np.arange(n) / (24 * 120)) + rng.normal(0, 0.5, n)
    )
    x = np.zeros(n)
    for k in range(1, n):
        x[k] = x[k - 1] + rate / 24 * (gain * 1.9 * (rts[k - 1] - 18.0) - x[k - 1])
    return hours, 19.5 + x + rng.normal(0, noise, n), t_out, rts


def fit(data, gain=GAIN, rate=RATE):
    hours, room, t_out, rts = data
    return fit_warmup(
        hours,
        room,
        t_out,
        rts,
        gain_c_per_c=gain,
        rate_per_day=rate,
        flow_per_rts=1.9,
        level_ref_c=19.5,
    )


def test_a_room_that_warms_as_the_model_says_gets_a_scale_of_one():
    result = fit(linear_room())
    assert result.scale == pytest.approx(1.0, abs=0.15)
    assert result.applied_scale == 1.0  # not different enough to act on
    assert result.windows > 300 and result.step_windows >= 30


def test_a_room_that_warms_faster_than_modelled_gets_a_larger_scale_that_is_used():
    result = fit(linear_room(gain=GAIN * 1.5))
    assert result.scale == pytest.approx(1.5, abs=0.2)
    assert result.applied_scale == pytest.approx(result.scale)


def test_a_slower_room_gets_a_smaller_scale_and_a_huge_difference_is_clipped():
    assert fit(linear_room(gain=GAIN * 0.6)).applied_scale == pytest.approx(
        0.6, abs=0.12
    )
    clipped = fit(linear_room(gain=GAIN * 5))
    assert clipped.applied_scale == pytest.approx(1.5)


def test_cooling_is_measured_separately_from_warming():
    result = fit(linear_room())
    assert result.cooling_response_c_per_c == pytest.approx(
        result.response_c_per_c, abs=0.04
    )


def test_without_setpoint_changes_the_warm_up_cannot_be_measured():
    hours, room, t_out, rts = linear_room()
    rts[:] = 18.0
    with pytest.raises(ModelFitError):
        fit((hours, room, t_out, rts))


def test_too_little_history_is_refused():
    with pytest.raises(ModelFitError):
        fit(linear_room(days=30))


def test_the_calibration_learns_the_warm_up_scale_with_the_other_models(settings):
    fitted = calibrator(settings, World(days=300), days=300).run(END)
    assert fitted.warmup is not None and fitted.warmup_error is None
    assert 0.5 <= fitted.warmup.applied_scale <= 1.5


def test_a_few_missing_hours_in_a_window_do_not_throw_the_window_away():
    hours, room, t_out, rts = linear_room()
    gappy = rts.copy()
    gappy[::7] = np.nan  # every seventh hour is missing
    complete = fit((hours, room, t_out, rts))
    with_gaps = fit((hours, room, t_out, gappy))
    assert with_gaps.windows >= 0.9 * complete.windows


def test_the_cooling_is_compared_with_the_model_too_and_a_faster_cooling_room_shows_it():
    same = fit(linear_room())
    assert same.cooling_scale == pytest.approx(1.0, abs=0.2) and same.cooling_se > 0
    # warms as modelled but cools faster: a room built that way
    hours, room, t_out, rts = linear_room()
    faster_down = room.copy()
    drop = np.cumsum(np.minimum(np.diff(rts, prepend=rts[0]), 0.0)) * 0.25
    faster_down += drop - np.mean(drop)
    assert fit((hours, faster_down, t_out, rts)).cooling_scale > same.cooling_scale
