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
        service.set_mode("shadow")
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


def closed_hours_now(service):
    snapshot = service.latest()
    return [
        h.valves_closed
        for h in snapshot.plan.hours
        if h.start <= service._now() < h.start + timedelta(hours=1)
    ]


def test_the_thermostats_are_held_at_the_floor_in_an_hour_the_plan_closes_the_loops(
    settings,
):
    service, ha, *_ = make(replace(settings, plan_closes_valves=True), evening=9.0)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert closed_hours_now(service) == [
        True
    ], "precondition: the plan closes this hour"
    assert ha.lk[LK] < LIMIT


def test_the_thermostats_go_back_to_the_limit_when_the_closed_hours_are_over(settings):
    service, ha, _, _, clock = make(
        replace(settings, plan_closes_valves=True), evening=9.0
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    clock[0] = datetime(2026, 10, 12, 21, 20, tzinfo=UTC)  # after the dear hours
    run_cycle(service)
    assert closed_hours_now(service) == [
        False
    ], "precondition: the plan keeps this hour open"
    assert ha.lk[LK] == LIMIT


def test_the_plan_never_closes_the_loops_when_the_switch_is_off(settings):
    service, ha, *_ = make(settings, evening=9.0)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert not any(h.valves_closed for h in service.latest().plan.hours)
    assert ha.lk[LK] == LIMIT


def test_the_hold_is_a_valid_half_degree_step_and_never_below_the_hours_floor(settings):
    """The LK thermostats take half-degree steps (target_temp_step 0.5); a floor of 17.85 (or 17.25) asked for
    as it is ends up as 17.8 or 17.7 in the thermostat."""
    service, ha, *_ = make(
        replace(
            settings,
            plan_closes_valves=True,
            extreme_floor_c=17.25,
            night_floor_c=17.25,
        ),
        evening=9.0,
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    now = service._now()
    hour = next(
        h
        for h in service.latest().plan.hours
        if h.start <= now < h.start + timedelta(hours=1)
    )
    assert hour.valves_closed and hour.floor_c == pytest.approx(17.25), "precondition"
    assert ha.lk[LK] * 2 == pytest.approx(round(ha.lk[LK] * 2))
    assert ha.lk[LK] >= hour.floor_c
