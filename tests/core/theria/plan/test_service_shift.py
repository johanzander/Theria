from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from core.theria.plan.plan_run import replay, run_from_dict, run_to_dict
from core.theria.plan.shift_rules import SalonStatus
from core.theria.plan.state_store import PlanStateStore

from .test_service import DAY_START, FakeHA, feed_periods, make_service

EVENING = DAY_START + timedelta(hours=17)
ROOM = "sensor.lk_bibliotek_temperature"


def shifting(settings):
    return replace(settings, rts_base_c=18.0)


def service_at(settings, moment, room=19.8, store=None, prices=None):
    service = make_service(
        shifting(settings),
        prices=prices or feed_periods(days=3),
        clock=lambda: moment,
        state_store=store,
    )
    reading = FakeHA.get_state

    def get_state(entity_id):
        if entity_id == ROOM:
            return {"state": str(room), "attributes": {}}
        return reading(service._ha, entity_id)

    service._ha.get_state = get_state
    return service


def test_the_evening_is_sold_when_the_house_can_take_it(settings):
    hours = service_at(settings, EVENING).run_once().plan.hours
    assert hours[0].reason["binding"] == "shift_sell"
    assert hours[0].rts_c == 17.0


def test_a_room_under_its_floor_does_not_end_the_sell(settings):
    hours = service_at(settings, EVENING, room=17.8).run_once().plan.hours
    assert hours[0].rts_c == 17.0


def test_the_ended_window_is_in_the_action_log(settings):
    service = service_at(settings, EVENING, room=17.8)
    service.run_once()
    assert any(e["kind"] == "shift" for e in service.action_log())


def test_the_salon_is_read_for_the_guard(settings):
    service = service_at(settings, EVENING)
    service.use_salon_status(lambda: SalonStatus(18.5, 19.0))
    hours = service.run_once().plan.hours
    assert hours[0].reason["guard"] == "salon_floor"


def test_a_salon_that_cannot_be_read_ends_the_sell(settings):
    service = service_at(settings, EVENING)

    def broken():
        raise KeyError("sensor")

    service.use_salon_status(broken)
    assert service.run_once().plan.hours[0].reason["guard"] == "salon_unknown"


def test_a_started_window_survives_a_restart(settings, tmp_path):
    store = PlanStateStore(str(tmp_path / "state.json"))
    service_at(settings, EVENING, store=store).run_once()
    flat = feed_periods(days=3, evening=1.0)
    hours = (
        service_at(settings, EVENING, store=store, prices=flat).run_once().plan.hours
    )
    assert hours[0].rts_c == 17.0


def test_a_run_replays_with_what_the_guards_read(settings):
    service = service_at(settings, EVENING, room=17.8)
    snapshot = service.run_once()
    captured = []
    service._plan_runs = type(
        "Runs", (), {"append_run": lambda self, run: captured.append(run)}
    )()
    service._shift_window = None
    service.run_once()
    run = run_from_dict(run_to_dict(captured[0]))
    assert run.inputs.shift.room_c == 17.8
    assert [h.rts_c for h in replay(run).hours] == [
        h.rts_c for h in snapshot.plan.hours
    ]


def test_a_shift_plan_is_recorded_and_shown_for_the_day(settings, tmp_path):
    from core.theria.plan.plan_record import PlanRecordStore

    service = make_service(
        shifting(settings),
        prices=feed_periods(days=3),
        clock=lambda: EVENING,
    )
    service._plan_record = PlanRecordStore(str(tmp_path / "record.jsonl"))
    service.run_once()
    view = service.day_view(EVENING.date())
    later = [h for h in view["hours"] if h.get("rts_planned") is not None]
    assert later


class FakeModel:
    def table(self, t_out, t24, t72, sun, wind, setpoints):
        return np.tile(np.asarray(setpoints, float) / 10.0, (len(t_out), 1))


class FakeForecaster:
    def run_due(self):
        return 0

    def heating_model(self):
        return FakeModel()


def test_the_plan_uses_the_energy_forecast_when_it_is_fitted(settings):
    service = service_at(settings, EVENING)
    service._energy_forecast = FakeForecaster()
    snapshot = service.run_once()
    first = snapshot.plan.hours[0]
    assert snapshot.model_status["energy"] == "forecast"
    assert first.energy_kwh == pytest.approx(first.rts_c / 10.0)


def test_the_plan_says_when_it_runs_without_the_energy_forecast(settings):
    snapshot = service_at(settings, EVENING).run_once()
    assert snapshot.model_status["energy"] == "pump model"
