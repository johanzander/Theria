from dataclasses import replace
from datetime import timedelta

from .test_service_active import make, run_cycle

RADIATOR = "climate.vantsidan"


def radiators(settings, state, action):
    service, ha, *_ = make(replace(settings, radiator_entities=(RADIATOR,)))
    original = ha.get_state

    def get_state(entity_id):
        if entity_id == RADIATOR:
            return {"state": state, "attributes": {"hvac_action": action}}
        return original(entity_id)

    ha.get_state = get_state
    return service


def test_a_radiator_calling_for_heat_does_not_change_what_the_plan_may_choose(settings):
    calling = radiators(settings, "heat", "heating")
    run_cycle(calling)
    idle = radiators(settings, "heat", "idle")
    run_cycle(idle)
    assert calling._effective_settings().rts_min_c < settings.rts_base_c
    assert [h.rts_c for h in calling.latest().plan.hours] == [
        h.rts_c for h in idle.latest().plan.hours
    ]


def test_while_a_radiator_calls_for_heat_the_pump_is_not_written_below_the_base(settings):
    idle = radiators(settings, "heat", "idle")
    idle.set_mode("active", confirm=True)
    run_cycle(idle)
    assert idle._ha.writes and min(idle._ha.writes) < settings.rts_base_c  # the peak is cut

    calling = radiators(settings, "heat", "heating")
    calling.set_mode("active", confirm=True)
    run_cycle(calling)
    assert calling._radiators["calling"]
    assert all(target >= settings.rts_base_c for target in calling._ha.writes)


def test_a_radiator_that_called_counts_for_an_hour_after_it_stops(settings):
    service = radiators(settings, "heat", "heating")
    run_cycle(service)
    first = service._now()
    service._ha.get_state = lambda e: (
        {"state": "heat", "attributes": {"hvac_action": "idle"}}
        if e == RADIATOR
        else {"state": "0", "attributes": {}}
    )
    service._now = lambda: first + timedelta(minutes=30)
    service._read_radiators()
    assert service._radiators == {"calling": True, "calling_now": False, "readable": 1}
    service._now = lambda: first + timedelta(minutes=61)
    service._read_radiators()
    assert service._radiators["calling"] is False


def test_when_the_radiators_are_satisfied_the_plan_may_cut(settings):
    service = radiators(settings, "heat", "idle")
    run_cycle(service)
    assert service._effective_settings().rts_min_c < settings.rts_base_c


def test_radiators_that_cannot_be_read_do_not_block_the_plan_but_are_reported(settings):
    service = radiators(settings, "unavailable", None)
    run_cycle(service)
    assert service._effective_settings().rts_min_c < settings.rts_base_c
    assert service.execution_status()["radiators"] == {
        "calling": False,
        "calling_now": False,
        "readable": 0,
    }


def test_no_radiators_configured_means_no_restriction(settings):
    service, *_ = make(settings)
    run_cycle(service)
    assert service.execution_status()["radiators"] is None
