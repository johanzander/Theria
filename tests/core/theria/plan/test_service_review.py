"""Findings of the independent review of the active-control layer."""

import asyncio
import threading
import time
from datetime import date

import pytest

from core.theria.exceptions import ConfigurationError, SetpointRestoreError
from core.theria.plan.executor import PlanExecutor
from core.theria.plan.state_store import PlanStateStore

from .test_service import feed_periods
from .test_service_active import (
    EVENING,
    WritableHA,
    good_days,
    make,
    run_cycle,
)


def test_a_failing_outcome_history_does_not_stop_the_control_cycle(settings):
    def broken(start, stop):
        raise RuntimeError("influx is down")

    service, ha, _, _, _ = make(settings, setpoint=18.0)
    service._outcome_loader = broken
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.writes  # the executor still ran
    assert service.execution_status()["reviewed_on"] is None  # and it will retry


def test_the_kill_switch_waits_for_a_running_control_cycle(settings):
    started, release = threading.Event(), threading.Event()

    def slow_history(start, stop):
        started.set()
        release.wait(5)
        return []

    service, ha, _, _, _ = make(settings, setpoint=17.0)
    service._outcome_loader = slow_history
    service.set_mode("active", confirm=True)
    service.run_once()
    control = threading.Thread(target=service._control)
    control.start()
    assert started.wait(5)
    switch = threading.Thread(target=lambda: service.set_mode("off"))
    switch.start()
    time.sleep(0.2)
    release.set()
    control.join(5)
    switch.join(5)
    assert service.mode == "off" and ha.setpoint == 18.0


def test_a_failed_restore_keeps_active_mode_and_says_so(settings, tmp_path):
    store = PlanStateStore(str(tmp_path / "state.json"))
    service, ha, _, _, _ = make(settings, setpoint=17.0, state_store=store)
    service.set_mode("active", confirm=True)
    ha.fail_writes = True
    with pytest.raises(SetpointRestoreError):
        service.set_mode("off")
    assert service.mode == "active"
    assert store.load().mode == "active"
    ha.fail_writes = False
    service.set_mode("off")
    assert ha.setpoint == 18.0 and service.mode == "off"


def test_the_base_setpoint_is_restored_before_the_mode_is_saved(settings, tmp_path):
    order = []
    store = PlanStateStore(str(tmp_path / "state.json"))
    service, ha, _, _, _ = make(settings, setpoint=17.0, state_store=store)
    service.set_mode("active", confirm=True)
    real_save, real_set = store.save, ha.set_temperature
    store.save = lambda state: (order.append("save"), real_save(state))[1]
    ha.set_temperature = lambda e, v: (order.append("write"), real_set(e, v))[1]
    service.set_mode("shadow")
    assert order == ["write", "save"]


def test_stopping_the_service_in_active_mode_puts_the_base_back(settings):
    service, ha, _, _, _ = make(settings, setpoint=17.0)
    service.set_mode("active", confirm=True)
    asyncio.run(service.stop())
    assert ha.setpoint == 18.0
    assert service.mode == "active"  # control resumes after a restart


def test_a_hand_change_is_noticed_even_when_nothing_was_written_yet(settings):
    ha = WritableHA(feed_periods(), 18.0)
    executor = PlanExecutor(ha, settings, now=lambda: EVENING)
    assert executor.execute(None).action == "kept"  # already at the base
    ha.setpoint = 19.0  # Johan turns it up
    assert executor.execute(None).action == "paused"
    assert ha.writes == []


def test_the_heater_check_uses_the_setpoint_that_is_really_set(settings):
    service, ha, _, _, _ = make(settings, setpoint=17.5)
    safety = service._safety
    service.set_mode("active", confirm=True)
    service.run_once()
    service._control()
    service._control()
    assert safety.calls[-1]["raised"] is False  # 17.5 is below the base, not raised
    ha.setpoint = 19.0
    service._control()
    assert safety.calls[-1]["raised"] is True


def test_trouble_while_at_the_base_is_not_caused_by_the_plan(settings):
    service, _, notifier, _, _ = make(settings, setpoint=18.0, trouble="stopped")
    service.set_mode("active", confirm=True)
    service.run_once()
    service._control()
    assert not notifier.created


def test_active_mode_is_refused_while_the_old_automation_is_on(settings):
    service, ha, _, _, _ = make(settings)
    ha.automation = "on"
    with pytest.raises(ConfigurationError):
        service.set_mode("active", confirm=True)
    ha.automation = "off"
    service.set_mode("active", confirm=True)
    ha.automation = "on"
    service.run_once()
    service._control()
    assert ha.writes == []  # it does not fight the automation
    assert "automation" in service.execution_status()["last_trouble"]


def test_a_daily_summary_of_yesterday_is_sent_when_the_day_is_judged(settings):
    service, _, notifier, _, _ = make(
        settings, outcomes=good_days(date(2026, 10, 7))  # the last day is 2026-10-11
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    summary = [m for i, m in notifier.created if i == "theria_plan_daily"]
    assert len(summary) == 1
    assert "2026-10-11" in summary[0] and "2 SEK" in summary[0]
    assert "room" in summary[0].lower()


def test_no_summary_is_sent_without_a_measured_yesterday(settings):
    service, _, notifier, _, _ = make(settings, outcomes=[])
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert not [i for i, _ in notifier.created if i == "theria_plan_daily"]
