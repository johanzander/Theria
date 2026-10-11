from dataclasses import replace
from datetime import UTC, date, datetime

import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.night_drift import NightDrift, NightObservation

from .test_service import FakeCalibrator, fitted_models, make_service

IN_THE_NIGHT = datetime(2026, 10, 12, 22, 20, tzinfo=UTC)  # 00:20 local
COLD_NIGHTS = NightDrift(
    -3.0,
    0.06,
    0.0,
    19.5,
    6,
    120,
    0.15,
    (NightObservation(date(2026, 10, 11), 6.0, -2.6, -2.64),),
)


@pytest.fixture
def night_settings(settings):
    return replace(settings, night_start_hour=0, night_end_hour=6)


def baseline_end_of_night(service):
    service.run_once()
    hours = service.latest().plan.baseline_hours
    return min(h.room_c_end for h in hours[:6])


def test_a_learned_night_cooling_shows_up_in_the_plan(night_settings):
    without = make_service(night_settings, clock=lambda: IN_THE_NIGHT)
    with_drift = make_service(night_settings, clock=lambda: IN_THE_NIGHT)
    with_drift._apply_models(replace(fitted_models(), night_drift=COLD_NIGHTS))
    assert baseline_end_of_night(with_drift) < baseline_end_of_night(without) - 1.0


def test_no_night_model_means_the_plan_is_what_it_was(night_settings):
    service = make_service(night_settings, clock=lambda: IN_THE_NIGHT)
    service._apply_models(fitted_models())  # night_drift is None
    assert service.model_status()["night"]["source"] == "none"


def test_the_model_status_reports_the_night_model_and_its_recent_nights(night_settings):
    service = make_service(night_settings, clock=lambda: IN_THE_NIGHT)
    service._apply_models(replace(fitted_models(), night_drift=COLD_NIGHTS))
    night = service.model_status()["night"]
    assert night["source"] == "fitted" and night["fit"].nights == 120


def test_a_rejected_night_fit_keeps_the_one_that_was_accepted(night_settings):
    good = replace(fitted_models(), night_drift=COLD_NIGHTS)
    rejected = replace(
        fitted_models(), night_drift=None, night_drift_error="ModelFitError: x"
    )
    calibrator = FakeCalibrator(good)
    service = make_service(night_settings, calibrator=calibrator)
    service.refit()
    calibrator.result = rejected
    service.refit()
    status = service.model_status()["night"]
    assert status["fit"] == COLD_NIGHTS and status["fit_error"] == "ModelFitError: x"


def test_a_failed_refit_does_not_touch_the_night_model(night_settings):
    calibrator = FakeCalibrator(replace(fitted_models(), night_drift=COLD_NIGHTS))
    service = make_service(night_settings, calibrator=calibrator)
    service.refit()
    calibrator.result, calibrator.error = None, ModelFitError("no data")
    with pytest.raises(ModelFitError):
        service.refit()
    assert service.model_status()["night"]["fit"] == COLD_NIGHTS


def drifting(recent_errors, rmse=0.2):
    from core.theria.plan.night_drift import NightObservation

    nights = tuple(
        NightObservation(date(2026, 10, 1 + i), 5.0, -0.5 + e, -0.5)
        for i, e in enumerate(recent_errors)
    )
    return NightDrift(-0.5, 0.05, -0.17, 19.5, 6, 100, rmse, nights)


def refit_with(service, drift):
    service._calibrator = FakeCalibrator(replace(fitted_models(), night_drift=drift))
    service.refit()


def night_alerts(notifier):
    return [m for i, m in notifier.created if i == "theria_plan_night_model"]


def test_a_night_model_that_keeps_missing_raises_an_alert(night_settings):
    service, _, notifier, _, _ = make_active(night_settings)
    refit_with(service, drifting([0.8, -0.7, 0.9, 0.8, -0.8, 0.7, 0.9]))
    assert (
        len(night_alerts(notifier)) == 1
        and "night" in night_alerts(notifier)[0].lower()
    )


def test_a_model_that_is_about_as_good_as_when_it_was_fitted_is_left_alone(
    night_settings,
):
    service, _, notifier, _, _ = make_active(night_settings)
    refit_with(service, drifting([0.1, -0.2, 0.15, -0.1, 0.2, -0.15, 0.1], rmse=0.2))
    assert night_alerts(notifier) == []


def test_the_alert_is_dismissed_when_the_model_is_good_again(night_settings):
    service, _, notifier, _, _ = make_active(night_settings)
    refit_with(service, drifting([0.8, -0.7, 0.9, 0.8]))
    refit_with(service, drifting([0.1, -0.1, 0.1, -0.1]))
    assert "theria_plan_night_model" in notifier.dismissed


def test_a_repeated_bad_refit_does_not_repeat_the_notification(night_settings):
    service, _, notifier, _, _ = make_active(night_settings)
    bad = drifting([0.8, -0.7, 0.9, 0.8])
    refit_with(service, bad)
    refit_with(service, bad)
    assert len(night_alerts(notifier)) == 1


def make_active(settings):
    from .test_service_active import make

    return make(settings, clock=lambda: IN_THE_NIGHT)


def test_the_plan_tells_how_long_before_comfort_starts_it_would_have_to_warm_up(
    night_settings,
):
    service, *_ = make_active(night_settings)
    service._apply_models(replace(fitted_models(), night_drift=COLD_NIGHTS))
    service.run_once()
    outlook = service.latest().comfort_outlook
    assert outlook is not None
    local_start = outlook.start.astimezone(
        __import__("zoneinfo").ZoneInfo("Europe/Stockholm")
    )
    assert local_start.hour == 6  # when the night ends
    assert outlook.deficit_c > 0 and outlook.lead_hours == pytest.approx(
        outlook.deficit_c / outlook.extra_rate_c_per_h
    )
    assert (
        outlook.lead_hours > 3
    )  # the floor is slow: a deficit takes hours to win back


def test_without_a_night_window_there_is_no_outlook(settings):
    from .test_service_active import make

    service, *_ = make(settings)
    service.run_once()
    assert service.latest().comfort_outlook is None


def warm(scale, applied=None):
    from core.theria.plan.warmup import WarmupFit

    return WarmupFit(
        scale,
        0.05,
        scale if applied is None else applied,
        0.15,
        0.14,
        0.2,
        1.0,
        0.1,
        380,
        90,
    )


def test_the_planner_warms_the_room_at_the_measured_speed(night_settings):
    slow, fast = make_active(night_settings)[0], make_active(night_settings)[0]
    slow._apply_models(
        replace(fitted_models(), night_drift=COLD_NIGHTS, warmup=warm(0.5))
    )
    fast._apply_models(
        replace(fitted_models(), night_drift=COLD_NIGHTS, warmup=warm(1.5))
    )
    slow.run_once()
    fast.run_once()
    assert fast.latest().comfort_outlook.extra_rate_c_per_h == pytest.approx(
        3 * slow.latest().comfort_outlook.extra_rate_c_per_h, rel=0.05
    )


def test_a_scale_that_is_not_significant_changes_nothing(night_settings):
    plain, noisy = make_active(night_settings)[0], make_active(night_settings)[0]
    plain._apply_models(replace(fitted_models(), night_drift=COLD_NIGHTS))
    noisy._apply_models(
        replace(fitted_models(), night_drift=COLD_NIGHTS, warmup=warm(1.1, applied=1.0))
    )
    plain.run_once()
    noisy.run_once()
    assert noisy.latest().comfort_outlook.extra_rate_c_per_h == pytest.approx(
        plain.latest().comfort_outlook.extra_rate_c_per_h
    )


def test_the_model_status_reports_the_warm_up_fit(night_settings):
    service, *_ = make_active(night_settings)
    service._apply_models(replace(fitted_models(), warmup=warm(1.2)))
    assert service.model_status()["warmup"]["fit"].scale == 1.2
    assert service.model_status()["warmup"]["source"] == "fitted"


def test_a_rejected_warm_up_fit_keeps_the_accepted_one(night_settings):
    good = replace(fitted_models(), warmup=warm(1.2))
    rejected = replace(fitted_models(), warmup=None, warmup_error="ModelFitError: x")
    calibrator = FakeCalibrator(good)
    service = make_service(night_settings, calibrator=calibrator)
    service.refit()
    calibrator.result = rejected
    service.refit()
    status = service.model_status()["warmup"]
    assert status["fit"].scale == 1.2 and status["fit_error"] == "ModelFitError: x"


def test_the_warm_up_scale_is_in_the_parameter_values(night_settings):
    service, *_ = make_active(night_settings)
    service._apply_models(replace(fitted_models(), warmup=warm(1.2)))
    params = service.parameters()
    assert params["warmup_scale"] == 1.2 and params["warmup_applied_scale"] == 1.2
