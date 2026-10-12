from dataclasses import replace
from datetime import timedelta

from core.theria.plan.night_drift import NightDrift
from core.theria.plan.stages import StageRules, StageState
from core.theria.plan.tuning import build_recommendations

from .test_stages import START, day, fresh

NIGHT = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 120, 0.1, ())


def recommend(settings, days=(), plan_hours=(), stage=None, night=NIGHT):
    return {
        r.key: r
        for r in build_recommendations(
            settings,
            list(days),
            list(plan_hours),
            stage or fresh(),
            night,
            StageRules(),
            START + timedelta(days=20),
        )
    }


def days_with_room_min(value, n=10):
    return [day(i, room_min=value) for i in range(n)]


def test_a_floor_the_room_never_comes_near_is_pointed_out(settings):
    rec = recommend(settings, days_with_room_min(21.0))["floor_c"]
    assert rec.status == "consider" and rec.suggestion is not None
    assert float(rec.suggestion.split()[0]) < settings.floor_c
    assert "21.0" in rec.observed


def test_a_floor_the_room_comes_close_to_is_left_alone(settings):
    rec = recommend(settings, days_with_room_min(settings.floor_c + 0.4))["floor_c"]
    assert rec.status == "ok" and rec.suggestion is None


def test_the_suggested_floor_never_goes_below_the_extreme_floor(settings):
    rec = recommend(settings, days_with_room_min(30.0))["floor_c"]
    assert float(rec.suggestion.split()[0]) >= settings.extreme_floor_c


def test_too_few_days_means_no_advice(settings):
    rec = recommend(settings, days_with_room_min(21.0, n=3))["floor_c"]
    assert rec.status == "no data" and rec.suggestion is None


def test_days_without_room_readings_do_not_count(settings):
    blind = [day(i, room_min=float("nan")) for i in range(10)]
    assert recommend(settings, blind)["floor_c"].status == "no data"


class Hour:
    def __init__(self, binding):
        self.reason = {"binding": binding}


def test_a_flow_cap_that_binds_often_is_reported(settings):
    hours = [Hour("flow_cap")] * 10 + [Hour("price_trade_off")] * 20
    rec = recommend(settings, plan_hours=hours)["max_flow_c"]
    assert rec.status == "consider" and "33" in rec.observed


def test_a_flow_cap_that_rarely_binds_is_fine(settings):
    hours = [Hour("flow_cap")] * 1 + [Hour("price_trade_off")] * 30
    assert recommend(settings, plan_hours=hours)["max_flow_c"].status == "ok"


def test_setpoint_limits_that_bind_for_most_hours_are_reported(settings):
    hours = [Hour("rts_min")] * 25 + [Hour("price_trade_off")] * 5
    rec = recommend(settings, plan_hours=hours)["rts_min_c"]
    assert rec.status == "consider" and "stage" in rec.reason.lower()


def test_the_stage_says_what_is_missing_before_it_widens(settings):
    days = [day(i) for i in range(2)]
    rec = recommend(settings, days)["stage"]
    assert rec.status == "ok" and "3 more" in rec.observed


def test_a_widest_stage_has_nothing_to_widen(settings):
    state = StageState(index=1, stage_started_on=START)
    assert recommend(settings, [day(0)], stage=state)["stage"].observed.startswith(
        "widest"
    )


def test_an_uncertain_night_model_is_called_uncertain(settings):
    few = replace(NIGHT, nights=20)
    assert recommend(settings, night=few)["night_model"].status == "consider"
    assert recommend(settings, night=None)["night_model"].status == "no data"
    assert recommend(settings)["night_model"].status == "ok"


def test_nothing_is_ever_applied_by_building_the_advice(settings):
    before = replace(settings)
    recommend(settings, days_with_room_min(21.0))
    assert settings == before
