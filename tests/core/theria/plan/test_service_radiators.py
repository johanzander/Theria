from dataclasses import replace

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


def test_while_a_radiator_calls_for_heat_the_plan_may_not_cut_below_the_base(settings):
    service = radiators(settings, "heat", "heating")
    run_cycle(service)
    assert service._effective_settings().rts_min_c == settings.rts_base_c
    assert all(h.rts_c >= settings.rts_base_c for h in service.latest().plan.hours[:3])


def test_when_the_radiators_are_satisfied_the_plan_may_cut(settings):
    service = radiators(settings, "heat", "idle")
    run_cycle(service)
    assert service._effective_settings().rts_min_c < settings.rts_base_c


def test_radiators_that_cannot_be_read_do_not_block_the_plan_but_are_reported(settings):
    service = radiators(settings, "unavailable", None)
    run_cycle(service)
    assert service._effective_settings().rts_min_c < settings.rts_base_c
    assert service.execution_status()["radiators"] == {"calling": False, "readable": 0}


def test_no_radiators_configured_means_no_restriction(settings):
    service, *_ = make(settings)
    run_cycle(service)
    assert service.execution_status()["radiators"] is None
