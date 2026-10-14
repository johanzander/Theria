import asyncio
from datetime import UTC, datetime

import pytest

from core.theria.exceptions import SetpointRestoreError

from .test_service_active import make, run_cycle

NOW = datetime(2026, 10, 12, 23, 20, tzinfo=UTC)
LK = "climate.lk_bibliotek_thermostat"
LIMIT = 22.0
BASE = 19.5  # the thermostats' setting before Theria, and when it does not control them


def test_the_thermostats_are_held_at_the_limit_in_active_mode(settings):
    service, ha, *_ = make(settings, clock=lambda: NOW)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.lk[LK] == LIMIT


def test_the_thermostats_are_not_touched_again_when_they_are_right(settings):
    service, ha, *_ = make(settings, clock=lambda: NOW)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    ha.lk_writes.clear()
    run_cycle(service)
    assert ha.lk_writes == []


def test_shadow_mode_never_touches_the_thermostats(settings):
    service, ha, *_ = make(settings, clock=lambda: NOW)
    run_cycle(service)
    assert ha.lk_writes == []


def test_leaving_active_mode_puts_the_thermostats_back_to_the_base(settings):
    service, ha, *_ = make(settings, clock=lambda: NOW)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    service.set_mode("shadow")
    assert ha.lk[LK] == BASE and ha.setpoint == 18.0


def test_a_failed_thermostat_restore_keeps_active_mode(settings):
    service, ha, *_ = make(settings, clock=lambda: NOW)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    ha.fail_lk_writes = True
    with pytest.raises(SetpointRestoreError):
        service.set_mode("off")
    assert service.mode == "active"


def test_stopping_the_service_restores_the_thermostats(settings):
    service, ha, *_ = make(settings, clock=lambda: NOW)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    asyncio.run(service.stop())
    assert ha.lk[LK] == BASE


def test_the_execution_status_reports_the_thermostat_action(settings):
    service, *_ = make(settings, clock=lambda: NOW)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert service.execution_status()["last_lk"].action == "wrote"
