from dataclasses import replace

from core.theria.plan.model_catalog import build_catalog
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS


def catalog(settings, **overrides):
    args = dict(
        settings=settings,
        pump=PRIOR_PUMP_PARAMS,
        pump_fitted=False,
        room=PRIOR_ROOM_PARAMS,
        room_fitted=False,
        night=None,
        warmup=None,
        morning_margin_c=0.0,
        disturbance_c_per_h=0.0,
        authority=0.5,
        skill_error_c_per_h=None,
        regulator_rise_c={},
        plan=None,
    )
    args.update(overrides)
    return build_catalog(**args)


def by_key(entries):
    return {e["key"]: e for e in entries}


def test_every_entry_says_what_it_is_where_it_comes_from_and_what_it_means(settings):
    entries = catalog(settings)
    assert entries
    for e in entries:
        assert e["group"] and e["label"] and e["description"], e["key"]
        assert e["source"] in {"fitted", "prior", "learned", "setting", "constant", "derived"}, e["key"]


def test_the_keys_are_unique(settings):
    keys = [e["key"] for e in catalog(settings)]
    assert len(keys) == len(set(keys))


def test_the_pump_room_sun_weights_and_limits_are_all_listed(settings):
    keys = set(by_key(catalog(settings)))
    for wanted in (
        "flow_intercept", "flow_slope", "flow_per_rts", "energy_intercept", "energy_per_outdoor",
        "energy_per_rts", "aux_onset_c", "aux_per_flow_c",
        "k_slab", "k_out", "tau_slab_h", "sun_swing_c_per_wm2", "sun_lag_h", "room_gain_c_per_c",
        "room_time_constant_days", "disturbance_c_per_h", "authority", "morning_margin_c",
        "comfort_cost_sek_per_c_hour", "setpoint_deviation_cost_sek_per_c_hour",
        "violation_cost_sek_per_c_hour", "switch_cost_sek_per_c", "end_credit_c",
        "min_saving_sek", "floor_mix_max_c", "lk_limit_c",
    ):
        assert wanted in keys, wanted


def test_fitted_and_prior_parameters_are_told_apart(settings):
    assert by_key(catalog(settings))["energy_per_rts"]["source"] == "prior"
    assert by_key(catalog(settings, pump_fitted=True))["energy_per_rts"]["source"] == "fitted"


def test_the_settings_show_the_value_in_use(settings):
    entries = by_key(catalog(replace(settings, comfort_cost_sek_per_c_hour=0.7)))
    assert entries["comfort_cost_sek_per_c_hour"]["value"] == 0.7
    assert entries["comfort_cost_sek_per_c_hour"]["source"] == "setting"


def test_a_learned_rise_is_listed_per_room(settings):
    entries = by_key(catalog(settings, regulator_rise_c={"bibliotek": 0.6}))
    assert entries["regulator_rise_bibliotek"]["value"] == 0.6
