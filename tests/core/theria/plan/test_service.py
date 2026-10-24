import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import (
    ConfigurationError,
    ModelFitError,
    PriceDataError,
    SensorError,
    WeatherDataError,
)
from core.theria.plan.calibration import FittedModels, PumpFit, RoomFit
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.prices import PricePeriod
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS, RoomParams
from core.theria.plan.service import PlanService
from core.theria.plan.state_store import PlanStateStore
from core.theria.plan.weather import WeatherHour

from .fits import daily_fit

NOW = datetime(2026, 10, 12, 12, 20, tzinfo=UTC)
DAY_START = datetime(2026, 10, 12, 0, tzinfo=UTC)


def feed_periods(days=2, base=1.0, evening=3.0) -> list[PricePeriod]:
    """What the Nord Pool feed answers: 15-minute prices from the start of the day, dear in the evening."""
    return [
        PricePeriod(
            DAY_START + timedelta(minutes=15 * q),
            evening if 16 <= (DAY_START + timedelta(minutes=15 * q)).hour < 20 else base,
        )
        for q in range(96 * days)
    ]


class FakeHA:
    def __init__(self, prices):
        self.prices = prices  # the feed the service reads
        self.reads = []

    def get_state(self, entity_id):
        self.reads.append(entity_id)
        states = {
            "sensor.outdoor": {"state": "6.0", "attributes": {}},
            "sensor.lk_bibliotek_temperature": {"state": "19.8", "attributes": {}},
            "sensor.heating_setpoint": {"state": "36.0", "attributes": {}},
            "sensor.radiator_forward": {"state": "36.0", "attributes": {}},
        }
        return states[entity_id]

    def get_climate_state(self, entity_id):
        return {"target_temperature": 18.0}


class FakeWeather:
    def __init__(self, hours=96, first=DAY_START - timedelta(hours=24)):
        self._hours = [
            WeatherHour(
                first + timedelta(hours=i),
                200.0 if 9 <= (first + timedelta(hours=i)).hour < 15 else 0.0,
                50.0,
                5.0,
            )
            for i in range(hours)
        ]

    def fetch(self):
        return self._hours


def make_service(
    settings,
    prices=None,
    weather=None,
    mode="shadow",
    clock=None,
    calibrator=None,
    state_store=None,
    sensor_guard=None,
    parameter_log=None,
    settings_manager=None,
):
    from dataclasses import replace

    ha = FakeHA(prices or feed_periods())
    return PlanService(
        ha,
        replace(settings, mode=mode),
        weather or FakeWeather(),
        price_periods=lambda: ha.prices,
        now=clock or (lambda: NOW),
        calibrator=calibrator,
        state_store=state_store,
        sensor_guard=sensor_guard,
        parameter_log=parameter_log,
        settings_manager=settings_manager,
    )


FITTED_PUMP = PumpParams(39.0, -0.8, 18.0, 1.9, 0.5, -0.1, 0.15)
FITTED_ROOM = RoomParams(4.0, 0.03, 0.04, 0.0004, 2.0, 0.0)


FITTED_ROOM_FIT = RoomFit(FITTED_ROOM, daily_fit())


def fitted_models(fitted_at=NOW, room_fit=FITTED_ROOM_FIT):
    return FittedModels(
        pump=PumpFit(FITTED_PUMP, 400, 0.5, 5000, 0.8),
        room=room_fit,
        room_error=None if room_fit else "ModelFitError: rejected",
        fitted_at=fitted_at,
        window_start=fitted_at - timedelta(days=365),
        window_end=fitted_at,
    )


class FakeCalibrator:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, 0

    def run(self, now):
        self.calls += 1
        if self.error:
            raise self.error
        return self.result


def test_run_once_plans_from_current_hour_to_end_of_known_prices(settings):
    service = make_service(settings)
    snapshot = service.run_once()
    hours = snapshot.plan.hours
    assert hours[0].start == datetime(2026, 10, 12, 12, tzinfo=UTC)
    assert len(hours) == 36  # 12:00 today to 24:00 tomorrow
    assert service.latest() is snapshot


def test_evening_price_hours_get_lower_setpoint_than_cheap_hours(settings):
    snapshot = make_service(settings).run_once()
    evening = [h.rts_c for h in snapshot.plan.hours if 16 <= h.start.hour < 20]
    other = [h.rts_c for h in snapshot.plan.hours if not 16 <= h.start.hour < 20]
    assert sum(evening) / len(evening) < sum(other) / len(other)


def test_off_mode_does_nothing_and_reads_nothing(settings):
    service = make_service(settings, mode="off")
    assert service.run_once() is None
    assert service._ha.reads == []


def test_set_mode_rejects_active_until_the_executor_exists(settings):
    service = make_service(settings)
    with pytest.raises(ConfigurationError):
        service.set_mode("active")
    service.set_mode("off")
    assert service.mode == "off"


def test_missing_tomorrow_prices_shortens_the_horizon(settings):
    snapshot = make_service(settings, prices=feed_periods(days=1)).run_once()
    assert len(snapshot.plan.hours) == 12


def test_weather_gap_raises_instead_of_guessing(settings):
    weather = FakeWeather(hours=30)  # ends before the horizon does
    with pytest.raises(WeatherDataError):
        make_service(settings, weather=weather).run_once()


def test_a_feed_without_a_complete_hour_raises_instead_of_guessing(settings):
    with pytest.raises(PriceDataError):
        make_service(settings, prices=feed_periods()[:3]).run_once()


def test_history_returns_only_recent_snapshots(settings):
    current = [NOW]
    service = make_service(settings, clock=lambda: current[0])
    service.run_once()
    assert len(service.history(hours=1)) == 1
    current[0] = NOW + timedelta(hours=2)
    assert service.history(hours=1) == []
    assert len(service.history(hours=3)) == 1


def test_unavailable_room_sensor_raises_sensor_error(settings):
    service = make_service(settings)
    original = service._ha.get_state

    def unavailable(entity_id):
        if entity_id == "sensor.lk_bibliotek_temperature":
            return {"state": "unavailable", "attributes": {}}
        return original(entity_id)

    service._ha.get_state = unavailable
    with pytest.raises(SensorError):
        service.run_once()


def test_models_are_priors_until_a_fit_is_accepted(settings):
    status = make_service(settings).model_status()
    assert status["pump"]["source"] == "prior" and status["room"]["source"] == "prior"
    assert status["fitted_at"] is None and status["last_fit_error"] is None


def test_refit_switches_to_fitted_models_and_plans_with_them(settings):
    service = make_service(settings, calibrator=FakeCalibrator(fitted_models()))
    before = service.run_once().plan.hours[0].flow_target_c
    service.refit()
    status = service.model_status()
    assert (
        status["pump"]["source"] == "fitted" and status["pump"]["params"] == FITTED_PUMP
    )
    assert status["room"]["params"] == FITTED_ROOM and status["fitted_at"] == NOW
    snapshot = service.run_once()
    assert snapshot.model_status == {"pump": "fitted", "room": "fitted"}
    assert snapshot.plan.hours[0].flow_target_c != before


def test_a_rejected_fit_keeps_the_previous_models_and_schedules_a_retry(settings):
    current = [NOW]
    calibrator = FakeCalibrator(error=ModelFitError("not enough heating hours"))
    service = make_service(settings, calibrator=calibrator, clock=lambda: current[0])
    assert service.refit_due()
    with pytest.raises(ModelFitError):
        service.refit()
    assert service.model_status()["pump"]["source"] == "prior"
    assert "ModelFitError" in service.model_status()["last_fit_error"]
    assert not service.refit_due()  # retry only after the retry interval
    current[0] = NOW + timedelta(hours=7)
    assert service.refit_due()


def test_a_successful_fit_is_not_repeated_before_the_refit_interval(settings):
    current = [NOW]
    service = make_service(
        settings, calibrator=FakeCalibrator(fitted_models()), clock=lambda: current[0]
    )
    service.refit()
    current[0] = NOW + timedelta(hours=23)
    assert not service.refit_due()
    current[0] = NOW + timedelta(hours=25)
    assert service.refit_due()


def test_refit_without_a_calibrator_is_a_configuration_error(settings):
    with pytest.raises(ConfigurationError):
        make_service(settings).refit()
    assert not make_service(settings).refit_due()


def test_mode_and_fitted_models_survive_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first = make_service(
        settings,
        calibrator=FakeCalibrator(fitted_models()),
        state_store=PlanStateStore(path),
    )
    first.refit()
    first.set_mode("off")
    second = make_service(settings, state_store=PlanStateStore(path), mode="shadow")
    assert second.mode == "off"
    assert second.model_status()["pump"]["params"] == FITTED_PUMP
    assert second.model_status()["room"]["params"] != PRIOR_ROOM_PARAMS


def test_scores_compare_earlier_predictions_with_later_readings(settings):
    current = [NOW]
    service = make_service(settings, clock=lambda: current[0])
    service.run_once()
    current[0] = NOW + timedelta(
        minutes=45
    )  # 13:05, five minutes after the plan hour ends
    service.run_once()
    scores = {s.lead_h: s for s in service.scores()}
    assert scores[1].n >= 1


def test_an_unexpected_refit_error_does_not_stop_the_planning_cycle(settings):
    calibrator = FakeCalibrator(error=RuntimeError("driver exploded"))
    service = make_service(settings, calibrator=calibrator)
    asyncio.run(service.cycle())  # must not raise
    assert service.latest() is not None  # planning still ran
    assert "RuntimeError" in service.model_status()["last_fit_error"]
    assert not service.refit_due()  # retried later, not on every cycle
    asyncio.run(service.cycle())
    assert calibrator.calls == 1


def test_a_rejected_room_fit_still_applies_the_pump_fit_and_keeps_the_room_prior(
    settings,
):
    service = make_service(
        settings, calibrator=FakeCalibrator(fitted_models(room_fit=None))
    )
    service.refit()
    status = service.model_status()
    assert status["pump"]["source"] == "fitted"
    assert (
        status["room"]["source"] == "prior"
        and status["room"]["params"] == PRIOR_ROOM_PARAMS
    )
    assert status["room"]["fit_error"] == "ModelFitError: rejected"
    assert service.run_once().model_status == {"pump": "fitted", "room": "prior"}


def test_a_later_rejection_keeps_the_room_fit_that_was_accepted_before(settings):
    results = [fitted_models(), fitted_models(room_fit=None)]

    class Sequence:
        def run(self, now):
            return results.pop(0)

    service = make_service(settings, calibrator=Sequence())
    service.refit()
    service.refit()
    status = service.model_status()
    assert (
        status["room"]["source"] == "fitted" and status["room"]["params"] == FITTED_ROOM
    )
    assert status["room"]["fit_error"] == "ModelFitError: rejected"


class FakeGuard:
    def __init__(self, value=None, error=None):
        self.value, self.error = value, error

    def room_temperature(self):
        if self.error:
            raise self.error
        return self.value


def test_room_temperature_comes_from_the_sensor_guard_when_there_is_one(settings):
    service = make_service(settings, sensor_guard=FakeGuard(19.0))
    assert service.run_once().initial.room_c == 19.0


def test_a_plan_is_not_made_when_no_comfort_sensor_is_healthy(settings):
    guard = FakeGuard(19.0)
    service = make_service(settings, sensor_guard=guard)
    service.run_once()
    guard.error = SensorError("no healthy comfort temperature sensor")
    with pytest.raises(SensorError):
        service.run_once()
    assert service.latest() is not None  # the previous plan stays available


def test_the_baseline_keeps_the_room_near_its_current_temperature(settings):
    # The model only knows how the flow temperature moves the room; the baseline (flat
    # setpoint) must therefore hold the room where it is, not drift to an invented equilibrium.
    snapshot = make_service(settings).run_once()
    drift = [
        abs(h.room_c_end - snapshot.initial.room_c)
        for h in snapshot.plan.baseline_hours
    ]
    assert max(drift) < 0.15
