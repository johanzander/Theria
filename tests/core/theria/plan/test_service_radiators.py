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


def test_a_radiator_calling_for_heat_does_not_change_what_the_plan_may_choose(settings):
    calling = radiators(settings, "heat", "heating")
    run_cycle(calling)
    idle = radiators(settings, "heat", "idle")
    run_cycle(idle)
    assert calling._effective_settings().rts_min_c < settings.rts_base_c
    assert [h.rts_c for h in calling.latest().plan.hours] == [
        h.rts_c for h in idle.latest().plan.hours
    ]


def test_a_radiator_calling_for_heat_does_not_stop_the_pump_from_being_cut(settings):
    calling = radiators(settings, "heat", "heating")
    calling.set_mode("active", confirm=True)
    run_cycle(calling)
    assert calling._ha.writes and min(calling._ha.writes) < settings.rts_base_c
