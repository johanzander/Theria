import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest

from core.theria.exceptions import ConfigurationError
from core.theria.plan.executor import PlanExecutor
from core.theria.plan.lk_executor import LKExecutor
from core.theria.plan.service import PlanService
from core.theria.plan.stages import StageState
from core.theria.plan.state_store import PlanState, PlanStateStore

from .test_service import FakeHA, FakeWeather, price_attributes
from .test_stages import day

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
        self.lk = {}  # LK thermostat setpoints (default 19.5)
        self.lk_writes = []
        self.fail_lk_writes = False

    def get_state(self, entity_id):
        if entity_id == "automation.set_room_temp_setpoint":
            return {"state": self.automation, "attributes": {}}
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

    def check(self, raised=False, deviated=True):
        self.calls.append({"raised": raised, "deviated": deviated})
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


def test_an_active_cycle_writes_the_plan_hour_within_the_stage(settings):
    service, ha, _, _, _ = make(settings)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert len(ha.writes) == 1
    assert (
        17.0 <= ha.writes[0] < 18.0
    )  # the first stage allows 1 C below the base; lower in the peak


def test_leaving_active_mode_puts_the_base_setpoint_back(settings):
    service, ha, _, _, _ = make(settings, setpoint=17.0)
    service.set_mode("active", confirm=True)
    service.set_mode("shadow")
    assert ha.writes == [18.0]


def test_trouble_restores_the_base_shrinks_the_stage_and_notifies(settings):
    service, ha, notifier, _, _ = make(
        settings, setpoint=17.0, trouble="compressor stopped"
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.writes[-1] == 18.0
    status = service.execution_status()
    assert status["stage"]["revert_until"] is not None
    assert notifier.created and "compressor stopped" in notifier.created[0][1]


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


def test_the_stage_is_judged_once_a_day_and_the_result_is_persisted(settings, tmp_path):
    store = PlanStateStore(str(tmp_path / "state.json"))
    store.save(
        PlanState(
            mode="active",
            models=None,
            stage=StageState(0, date(2026, 10, 6), None),
            stage_checked_on=None,
        )
    )
    service, _, notifier, calls, _ = make(
        settings,
        state_store=store,
        outcomes=good_days(date(2026, 10, 6)),
    )
    run_cycle(service)
    run_cycle(service)
    assert len(calls) == 1  # once per day
    assert service.execution_status()["stage"]["index"] == 1
    saved = PlanStateStore(str(tmp_path / "state.json")).load()
    assert saved.stage.index == 1 and saved.stage_checked_on == date(2026, 10, 12)
    assert any(
        "widen" in message or "bredd" in message.lower()
        for _, message in notifier.created
    )


def test_active_mode_and_the_stage_survive_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first, *_ = make(settings, state_store=PlanStateStore(path))
    first.set_mode("active", confirm=True)
    second, *_ = make(settings, state_store=PlanStateStore(path))
    assert second.mode == "active"
    assert second.execution_status()["stage"]["index"] == 0
