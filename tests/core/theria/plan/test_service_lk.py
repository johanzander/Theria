import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

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


def extreme_hours_now(service):
    snapshot = service.latest()
    return [
        h.extreme
        for h in snapshot.plan.hours
        if h.start <= service._now() < h.start + timedelta(hours=1)
    ]


def test_the_thermostats_are_held_at_the_extreme_floor_in_an_extreme_price_hour(settings):
    service, ha, *_ = make(replace(settings, valves_closed_when_extreme=True), evening=9.0)  # now is in the dear evening hours
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert extreme_hours_now(service) == [True], "precondition: this hour is extreme"
    assert ha.lk[LK] == settings.extreme_floor_c


def test_the_thermostats_go_back_to_the_limit_when_the_extreme_hour_is_over(settings):
    service, ha, _, _, clock = make(replace(settings, valves_closed_when_extreme=True), evening=9.0)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    clock[0] = datetime(2026, 10, 12, 21, 20, tzinfo=UTC)  # after the dear hours
    run_cycle(service)
    assert extreme_hours_now(service) == [False], "precondition: this hour is not extreme"
    assert ha.lk[LK] == LIMIT


def test_the_valves_stay_open_in_an_extreme_hour_when_the_switch_is_off(settings):
    service, ha, *_ = make(settings, evening=9.0)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert extreme_hours_now(service) == [True], "precondition: this hour is extreme"
    assert ha.lk[LK] == LIMIT
