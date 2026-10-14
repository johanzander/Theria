from datetime import timedelta

from core.theria.plan.disturbance import Innovation
from core.theria.plan.state_store import PlanStateStore

from .test_service_active import EVENING, make


def good_evidence(service, hours=48):
    for k in range(hours * 4):
        wobble = 0.3 if k % 2 == 0 else -0.3
        service._skill.add(Innovation(EVENING - timedelta(minutes=15 * k), wobble, True))


def test_without_evidence_the_plan_has_half_the_swing_around_the_base(settings):
    service, *_ = make(settings)
    effective = service._effective_settings()
    assert (effective.rts_min_c, effective.rts_max_c) == (17.0, 19.0)


def test_a_model_that_has_predicted_well_gives_the_plan_the_full_range(settings):
    service, *_ = make(settings)
    good_evidence(service)
    effective = service._effective_settings()
    assert (effective.rts_min_c, effective.rts_max_c) == (
        settings.rts_min_c,
        settings.rts_max_c,
    )
    assert service.parameters()["authority"] == 1.0


def test_the_evidence_survives_a_restart(settings, tmp_path):
    path = str(tmp_path / "state.json")
    first, *_ = make(settings, state_store=PlanStateStore(path))
    good_evidence(first)
    first._save_state()
    second, *_ = make(settings, state_store=PlanStateStore(path))
    assert second.parameters()["authority"] == 1.0
