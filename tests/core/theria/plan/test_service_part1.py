from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from core.theria.exceptions import SetpointRestoreError

from .test_service_active import EVENING, make, run_cycle

LK = "climate.lk_bibliotek_thermostat"
NOON = datetime(2026, 10, 12, 10, 20, tzinfo=UTC)


def shifting(settings):
    return replace(
        settings, plan_strategy="heat_shift", rts_base_c=18.0, plan_closes_valves=True
    )


def test_a_sell_closes_the_thermostat_in_active_mode(settings):
    service, ha, *_ = make(shifting(settings), clock=lambda: EVENING)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert service.latest().plan.hours[0].thermostat == "closing"
    assert ha.lk[LK] == 19.0
    assert ha.setpoint == 17.0


def test_a_sell_writes_nothing_in_shadow_mode(settings):
    service, ha, *_ = make(shifting(settings), clock=lambda: EVENING)
    run_cycle(service)
    assert service.latest().plan.hours[0].thermostat == "closing"
    assert ha.lk_writes == []


def test_outside_a_window_the_thermostat_stands_at_the_base(settings):
    service, ha, *_ = make(shifting(settings), clock=lambda: NOON)
    ha.lk[LK] = 19.0
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.lk[LK] == settings.lk_base_c


def test_a_setting_can_keep_the_loops_out_of_the_plan(settings):
    service, ha, *_ = make(
        replace(shifting(settings), plan_closes_valves=False), clock=lambda: EVENING
    )
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert service.latest().plan.hours[0].thermostat == "normal"
    assert ha.setpoint == 17.0


def test_a_room_that_failed_to_restore_gets_no_window(settings):
    service, ha, *_ = make(shifting(settings), clock=lambda: EVENING)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    ha.fail_lk_writes = True
    with pytest.raises(SetpointRestoreError):
        service.set_mode("shadow")
    ha.fail_lk_writes = False
    run_cycle(service)
    assert ha.lk[LK] == settings.lk_base_c


def test_the_optimizer_keeps_its_limit(settings):
    service, ha, *_ = make(settings, clock=lambda: NOON)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.lk[LK] == 22.0


def test_a_stale_plan_gets_no_window(settings):
    moment = [EVENING]
    service, *_ = make(shifting(settings), clock=lambda: moment[0])
    service.set_mode("active", confirm=True)
    run_cycle(service)
    moment[0] = EVENING + timedelta(hours=2)
    assert service._shift_holds() == ({LK: settings.lk_base_c}, None)
