import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.executor import PlanExecutor
from core.theria.plan.lk_executor import LKExecutor
from core.theria.plan.service import PlanService
from core.theria.plan.state_store import PlanState, PlanStateStore

from .days import day
from .test_service import FakeHA, FakeWeather, price_attributes

EVENING = datetime(
    2026, 10, 12, 16, 20, tzinfo=UTC
)  # inside the expensive hours of the fake prices


class WritableHA(FakeHA):
    def __init__(self, attrs, setpoint=18.0):
        super().__init__(attrs)
        self.setpoint = setpoint
        self.writes = []
        self.automation = "off"  # the old setpoint automation
        self.fail_writes = False
        self.pump_states = {
            "binary_sensor.switch_valve_1": "off",
            "sensor.add_heat_status": "0",
        }
        self.lk = {}  # LK thermostat setpoints (default 19.5)
        self.lk_writes = []
        self.fail_lk_writes = False

    def get_state(self, entity_id):
        if entity_id == "automation.set_room_temp_setpoint":
            return {"state": self.automation, "attributes": {}}
        if entity_id in self.pump_states:
            return {"state": self.pump_states[entity_id], "attributes": {}}
        return super().get_state(entity_id)

    def get_climate_state(self, entity_id):
        if entity_id.startswith("climate.lk"):
            return {"target_temperature": self.lk.get(entity_id, 19.5)}
        return {"target_temperature": self.setpoint}

    def set_temperature(self, entity_id, temperature):
        if entity_id.startswith("climate.lk"):
            if self.fail_lk_writes:
                raise RuntimeError("LK unreachable")
            self.lk_writes.append(temperature)
            self.lk[entity_id] = temperature
            return
        if self.fail_writes:
            raise RuntimeError("HA unreachable")
        self.writes.append(temperature)
        self.setpoint = temperature


class FakeSafety:
    def __init__(self, trouble=None):
        self.trouble = trouble
        self.calls = []

    def check(self, raised=False):
        self.calls.append({"raised": raised})
        return self.trouble


class FakeNotifier:
    def __init__(self):
        self.created, self.dismissed = [], []

    def notify(self, notification_id, title, message):
        self.created.append((notification_id, message))

    def dismiss(self, notification_id):
        self.dismissed.append(notification_id)


def make(
    settings,
    *,
    setpoint=18.0,
    trouble=None,
    clock=None,
    state_store=None,
    outcomes=None,
    event_log=None,
    actuals_loader=None,
):
    ha = WritableHA(price_attributes(), setpoint)
    current = [EVENING]
    now = clock or (lambda: current[0])
    notifier = FakeNotifier()
    calls = []

    def loader(start, stop):
        calls.append((start, stop))
        return outcomes or []

    service = PlanService(
        ha,
        replace(settings, mode="shadow"),
        FakeWeather(),
        now=now,
        executor=PlanExecutor(ha, settings, now=now),
        lk_executor=LKExecutor(ha, settings, now=now),
        safety=FakeSafety(trouble),
        notifier=notifier,
        outcome_loader=loader,
        state_store=state_store,
        event_log=event_log,
        actuals_loader=actuals_loader,
    )
    return service, ha, notifier, calls, current


def run_cycle(service):
    asyncio.run(service.cycle())


def test_shadow_mode_never_writes(settings):
    service, ha, _, _, _ = make(settings)
    run_cycle(service)
    assert service.latest() is not None and ha.writes == []


def test_active_mode_needs_explicit_confirmation(settings):
    service, *_ = make(settings)
    with pytest.raises(ConfigurationError):
        service.set_mode("active")
    service.set_mode("active", confirm=True)
    assert service.mode == "active"


def test_an_active_cycle_writes_the_plan_hour_within_the_setpoint_range(settings):
    service, ha, _, _, _ = make(settings)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert len(ha.writes) == 1
    assert settings.rts_min_c <= ha.writes[0] < 18.0  # lower in the peak


def test_leaving_active_mode_puts_the_base_setpoint_back(settings):
    service, ha, _, _, _ = make(settings, setpoint=17.0)
    service.set_mode("active", confirm=True)
    service.set_mode("shadow")
    assert ha.writes == [18.0]


def test_trouble_is_reported_and_notified_and_the_plan_keeps_running(settings):
    service, _, notifier, _, _ = make(
        settings, setpoint=17.0, trouble="electric heater"
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert "electric heater" in service.execution_status()["last_trouble"]
    assert notifier.created and "electric heater" in notifier.created[0][1]


def test_the_executor_still_runs_when_planning_fails_and_the_plan_gets_old(settings):
    service, ha, _, _, current = make(settings)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.writes and ha.setpoint != 18.0
    service._weather = FakeWeather(
        hours=2
    )  # the forecast disappears: planning now fails
    current[0] += timedelta(hours=1)
    run_cycle(service)
    assert ha.setpoint == 18.0  # old plan: back to the base


def good_days(first, n=5):
    return [replace(day(i), day=first + timedelta(days=i)) for i in range(n)]


def test_the_days_are_reviewed_once_a_day_and_the_result_is_persisted(
    settings, tmp_path
):
    store = PlanStateStore(str(tmp_path / "state.json"))
    store.save(
        PlanState(
            mode="active",
            models=None,
            reviewed_on=None,
        )
    )
    service, _, notifier, calls, _ = make(
        settings,
        state_store=store,
        outcomes=good_days(date(2026, 10, 6), n=6),
    )
    run_cycle(service)
    run_cycle(service)
    assert len(calls) == 1  # once per day
    saved = PlanStateStore(str(tmp_path / "state.json")).load()
    assert saved.reviewed_on == date(2026, 10, 12)
    assert any("saving" in message for _, message in notifier.created)


def test_the_last_trouble_survives_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first, *_ = make(
        settings,
        setpoint=17.0,
        trouble="electric heater",
        state_store=PlanStateStore(path),
    )
    first.set_mode("active", confirm=True)
    run_cycle(first)
    second, *_ = make(settings, state_store=PlanStateStore(path))
    assert "electric heater" in second.execution_status()["last_trouble"]


def test_active_mode_survives_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first, *_ = make(settings, state_store=PlanStateStore(path))
    first.set_mode("active", confirm=True)
    second, *_ = make(settings, state_store=PlanStateStore(path))
    assert second.mode == "active"


def test_stopping_and_starting_active_mode_ends_a_pause_after_a_hand_change(settings):
    service, ha, _, _, current = make(settings)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    written = ha.setpoint
    ha.setpoint = written + 1.0  # Johan turns it up by hand
    current[0] += timedelta(minutes=15)
    run_cycle(service)
    assert service.execution_status()["last_execution"].action == "paused"

    service.set_mode("shadow")
    service.set_mode("active", confirm=True)
    current[0] += timedelta(minutes=15)
    run_cycle(service)

    assert service.execution_status()["last_execution"].action != "paused"
