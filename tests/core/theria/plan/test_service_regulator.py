from dataclasses import replace
from datetime import timedelta

from core.theria.plan.settings import RoomControl
from core.theria.plan.state_store import PlanStateStore

from .test_service_active import EVENING, make, run_cycle

LK = "climate.lk_bibliotek_thermostat"
CONTROL = RoomControl("bibliotek", LK, ("sensor.lk_bibliotek_temperature",))


def reading(ha, value):
    original = ha.get_state

    def get_state(entity_id):
        if entity_id == "sensor.lk_bibliotek_temperature":
            return {"state": str(value), "attributes": {}}
        return original(entity_id)

    ha.get_state = get_state


def regulated(settings, active=True, **kwargs):
    return make(
        replace(settings, room_controls=(CONTROL,), regulator_active=active), **kwargs
    )


def test_a_room_over_its_limit_gets_its_thermostat_held_below_the_room(settings):
    service, ha, _, _, clock = regulated(settings)
    reading(ha, 22.4)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.lk[LK] == 21.5  # below the room: the valves close
    assert ("regulator", "closed") in [(e["kind"], e["action"]) for e in service.action_log()]


def test_the_thermostat_goes_back_to_the_limit_when_the_room_has_cooled(settings):
    service, ha, _, _, clock = regulated(settings)
    reading(ha, 22.4)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    reading(ha, 21.0)
    clock[0] = EVENING + timedelta(minutes=15)
    run_cycle(service)
    assert ha.lk[LK] == 22.0


def test_a_regulator_that_is_not_active_logs_what_it_would_do_and_holds_nothing(settings):
    service, ha, _, _, clock = regulated(settings, active=False)
    reading(ha, 22.4)
    service.set_mode("active", confirm=True)
    run_cycle(service)
    assert ha.lk[LK] == 22.0
    entry = next(e for e in service.action_log() if e["kind"] == "regulator")
    assert entry["detail"].startswith("would")


def test_shadow_mode_measures_but_never_writes(settings):
    service, ha, _, _, clock = regulated(settings)
    reading(ha, 22.4)
    run_cycle(service)
    assert ha.lk_writes == []
    status = service.execution_status()["regulator"]
    assert status[0].room == "bibliotek" and status[0].state == "closed"


def test_the_learned_rise_survives_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first, ha, *_ = regulated(settings, state_store=PlanStateStore(path))
    first._regulator._rooms["bibliotek"].rise_c = 0.6
    first._regulator._rooms["bibliotek"].events = 2
    first._save_state()
    second, *_ = regulated(settings, state_store=PlanStateStore(path))
    assert second._regulator._rooms["bibliotek"].rise_c == 0.6


def test_the_plan_aims_below_the_limit_by_the_rise_the_floor_adds_after_the_valves_close(
    settings,
):
    service, *_ = regulated(settings)
    assert service._effective_settings().ceiling_c == 22.0
    service._regulator._rooms["bibliotek"].rise_c = 0.6
    assert service._effective_settings().ceiling_c == 21.4
    service._regulator._rooms["bibliotek"].rise_c = 2.0
    assert service._effective_settings().ceiling_c >= settings.target_c + 0.5
