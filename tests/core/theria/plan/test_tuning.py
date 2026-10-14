from dataclasses import replace

from core.theria.plan.night_drift import NightDrift
from core.theria.plan.tuning import build_recommendations

from .days import day

NIGHT = NightDrift(-0.5, 0.06, -0.2, 19.5, 6, 120, 0.1, ())


def recommend(settings, days=(), plan_hours=(), night=NIGHT, authority=1.0):
    return {
        r.key: r
        for r in build_recommendations(
            settings,
            list(days),
            list(plan_hours),
            night,
            authority=authority,
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
    def __init__(self, binding, flow_target_c=36.0):
        self.reason = {"binding": binding}
        self.flow_target_c = flow_target_c


def hours_with_flow(n_over, n_under, cap):
    return [Hour("price_trade_off", cap + 4.0)] * n_over + [Hour("price_trade_off", cap - 4.0)] * n_under


def test_hours_where_the_pump_asks_for_more_than_the_floor_gets_are_reported(settings):
    hours = hours_with_flow(10, 20, settings.floor_mix_max_c)
    rec = recommend(settings, plan_hours=hours)["floor_mix_max_c"]
    assert rec.status == "consider" and "33" in rec.observed
    assert "nothing" in rec.reason.lower()  # lowering the pump setpoint costs the floor nothing there


def test_a_floor_cap_that_rarely_matters_is_fine(settings):
    hours = hours_with_flow(1, 30, settings.floor_mix_max_c)
    assert recommend(settings, plan_hours=hours)["floor_mix_max_c"].status == "ok"


def test_setpoint_limits_that_bind_for_most_hours_are_reported_at_full_authority(settings):
    hours = [Hour("rts_min")] * 25 + [Hour("price_trade_off")] * 5
    rec = recommend(settings, plan_hours=hours)["rts_min_c"]
    assert rec.status == "consider" and "electric heater" in rec.reason.lower()


def test_while_the_plan_has_only_part_of_the_range_the_limits_are_not_judged(settings):
    hours = [Hour("rts_min")] * 25 + [Hour("price_trade_off")] * 5
    rec = recommend(settings, plan_hours=hours, authority=0.5)["rts_min_c"]
    assert rec.status == "no data" and "50 %" in rec.observed


def test_the_old_flow_cap_advice_is_gone_the_shunts_protect_the_floor(settings):
    assert "max_flow_c" not in recommend(settings)


def test_an_uncertain_night_model_is_called_uncertain(settings):
    few = replace(NIGHT, nights=20)
    assert recommend(settings, night=few)["night_model"].status == "consider"
    assert recommend(settings, night=None)["night_model"].status == "no data"
    assert recommend(settings)["night_model"].status == "ok"


def test_nothing_is_ever_applied_by_building_the_advice(settings):
    before = replace(settings)
    recommend(settings, days_with_room_min(21.0))
    assert settings == before
