import asyncio
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from core.theria.exceptions import SetpointRestoreError

from .test_service_active import make, run_cycle

CHEAP_BEFORE_PEAK = datetime(2026, 10, 12, 23, 20, tzinfo=UTC)
PEAK_PRICE = 8.0  # a clear spread: the plan pre-heats before it and cuts through it


def preheating(settings):
    """Settings for which the plan pre-heats at CHEAP_BEFORE_PEAK: the room (19.8) is under the target."""
    return replace(settings, target_c=20.5)

LK = "climate.lk_bibliotek_thermostat"


def test_the_thermostats_are_raised_while_the_plan_preheats(settings):
    service, ha, _, _, _ = make(preheating(settings), clock=lambda: CHEAP_BEFORE_PEAK, evening=PEAK_PRICE)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.setpoint > 18.0, "precondition: the plan pre-heats at this hour"
    assert ha.lk[LK] == 21.0


def test_they_stay_at_the_base_when_the_plan_coasts(settings):
    service, ha, _, _, _ = make(settings)  # 16:20, the expensive hours
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.setpoint < 18.0 and ha.lk_writes == []


def test_shadow_mode_never_touches_the_thermostats(settings):
    service, ha, _, _, _ = make(preheating(settings), clock=lambda: CHEAP_BEFORE_PEAK, evening=PEAK_PRICE)
    run_cycle(service)
    assert ha.lk_writes == []


def test_leaving_active_mode_lowers_the_thermostats_again(settings):
    service, ha, _, _, _ = make(preheating(settings), clock=lambda: CHEAP_BEFORE_PEAK, evening=PEAK_PRICE)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    service.set_mode("shadow")
    assert ha.lk[LK] == 19.5 and ha.setpoint == 18.0


def test_a_failed_thermostat_restore_keeps_active_mode(settings):
    service, ha, _, _, _ = make(preheating(settings), clock=lambda: CHEAP_BEFORE_PEAK, evening=PEAK_PRICE)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    ha.fail_lk_writes = True
    with pytest.raises(SetpointRestoreError):
        service.set_mode("off")
    assert service.mode == "active"


def test_stopping_the_service_restores_the_thermostats(settings):
    service, ha, _, _, _ = make(preheating(settings), clock=lambda: CHEAP_BEFORE_PEAK, evening=PEAK_PRICE)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    asyncio.run(service.stop())
    assert ha.lk[LK] == 19.5


def test_the_execution_status_reports_the_thermostat_action(settings):
    service, _, _, _, _ = make(preheating(settings), clock=lambda: CHEAP_BEFORE_PEAK, evening=PEAK_PRICE)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert service.execution_status()["last_lk"].action == "wrote"
