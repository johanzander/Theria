from dataclasses import replace

import pytest

from core.theria.plan.calibration import RoomFit
from core.theria.plan.daily_room import DailyRoomFit, room_params_from_fit
from core.theria.plan.night_drift import NightDrift
from core.theria.plan.parameter_log import ParameterLog
from core.theria.plan.trim import (
    MAX_APPLIED_SCALE_STEP,
    MAX_GAIN_STEP,
    MAX_INTERCEPT_STEP_C,
    MAX_RATE_STEP_PER_DAY,
    limit_step,
    trim_night,
    trim_room,
    trim_warmup,
)
from core.theria.plan.warmup import WarmupFit

NIGHT = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 120, 0.1, ())


def room_fit(gain, rate):
    response = DailyRoomFit(gain, rate, 1 / rate, (), 100, {}, {})
    return RoomFit(room_params_from_fit(response), response)


WARMUP = WarmupFit(1.0, 0.1, 1.0, 0.4, 0.4, 0.5, 1.0, 0.1, 300, 40)


def test_a_step_within_the_limit_is_taken_whole():
    assert limit_step(1.0, 1.04, 0.05) == pytest.approx(1.04)


def test_a_larger_step_is_cut_at_the_limit_in_the_right_direction():
    assert limit_step(1.0, 2.0, 0.05) == pytest.approx(1.05)
    assert limit_step(1.0, -2.0, 0.05) == pytest.approx(0.95)


def test_a_first_fit_has_nothing_to_be_limited_against():
    assert trim_night(None, NIGHT) == (NIGHT, [])


def test_a_jump_in_the_night_model_is_taken_in_steps_and_reported():
    jumped = replace(NIGHT, intercept_c=-1.5)
    trimmed, notes = trim_night(NIGHT, jumped)
    assert trimmed.intercept_c == pytest.approx(-0.5 - MAX_INTERCEPT_STEP_C)
    assert trimmed.slope_c_per_c == NIGHT.slope_c_per_c
    assert len(notes) == 1 and "night intercept" in notes[0]


def test_the_rest_of_the_new_fit_is_kept():
    refit = replace(NIGHT, nights=150, rmse_c=0.3)
    trimmed, notes = trim_night(NIGHT, refit)
    assert trimmed.nights == 150 and trimmed.rmse_c == 0.3 and notes == []


def test_the_warm_up_scale_the_planner_uses_moves_in_steps():
    new = replace(WARMUP, applied_scale=1.6)
    trimmed, notes = trim_warmup(WARMUP, new)
    assert trimmed.applied_scale == pytest.approx(1.0 + MAX_APPLIED_SCALE_STEP)
    assert trimmed.scale == new.scale  # the measurement itself is shown as it is
    assert notes and "warm-up" in notes[0]


# ---- in the service

from .test_service import FakeCalibrator, fitted_models, make_service  # noqa: E402


def refit_with(settings, first_night, second_night, parameter_log=None):
    calibrator = FakeCalibrator(replace(fitted_models(), night_drift=first_night))
    service = make_service(settings, calibrator=calibrator, parameter_log=parameter_log)
    service.refit()
    calibrator.result = replace(fitted_models(), night_drift=second_night)
    service.refit()
    return service


def test_a_refit_that_jumps_moves_the_planner_only_one_step_and_says_so(
    settings, tmp_path
):
    log = ParameterLog(str(tmp_path / "p.jsonl"))
    service = refit_with(settings, NIGHT, replace(NIGHT, intercept_c=-1.5), log)
    assert service.parameters()["night_intercept_c"] == pytest.approx(
        -0.5 - MAX_INTERCEPT_STEP_C
    )
    assert "night intercept" in log.read()[-1]["note"]


def test_the_first_fit_is_taken_as_it_is(settings):
    calibrator = FakeCalibrator(
        replace(fitted_models(), night_drift=replace(NIGHT, intercept_c=-1.5))
    )
    service = make_service(settings, calibrator=calibrator)
    service.refit()
    assert service.parameters()["night_intercept_c"] == -1.5


def test_the_room_response_moves_in_steps_and_the_hourly_parameters_follow():
    trimmed, notes = trim_room(room_fit(0.45, 0.20), room_fit(0.80, 0.20))
    assert trimmed.response.gain_c_per_c == pytest.approx(0.45 + MAX_GAIN_STEP)
    assert trimmed.params == room_params_from_fit(trimmed.response)
    assert len(notes) == 1 and "room gain" in notes[0]


def test_a_faster_room_is_limited_too_and_the_time_constant_stays_consistent():
    trimmed, notes = trim_room(room_fit(0.45, 0.20), room_fit(0.45, 0.50))
    assert trimmed.response.rate_per_day == pytest.approx(0.20 + MAX_RATE_STEP_PER_DAY)
    assert trimmed.response.time_constant_days == pytest.approx(
        1 / trimmed.response.rate_per_day
    )
    assert "room rate" in notes[0]


def test_a_small_change_in_the_room_response_is_left_alone():
    new = room_fit(0.46, 0.21)
    assert trim_room(room_fit(0.45, 0.20), new) == (new, [])
