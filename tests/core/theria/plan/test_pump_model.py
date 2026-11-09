import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.pump_model import (
    PRIOR_PUMP_PARAMS,
    PumpParams,
    energy_kwh_per_hour,
    fit_pump_params,
    fit_rts_levels,
    flow_target,
)


def test_prior_reproduces_measured_curve_and_room_setpoint_gain():
    p = PRIOR_PUMP_PARAMS
    assert flow_target(p, 12.0, p.rts_ref) == pytest.approx(32.64)
    assert flow_target(p, 5.0, 19.0) - flow_target(p, 5.0, 18.0) == pytest.approx(1.9)


def test_energy_is_higher_with_higher_setpoint_and_never_negative():
    p = PRIOR_PUMP_PARAMS
    assert energy_kwh_per_hour(p, 5.0, 20.0) > energy_kwh_per_hour(p, 5.0, 16.0)
    assert float(energy_kwh_per_hour(p, 40.0, 13.0)) == 0.0


def make_data(n=200):
    rng = np.random.default_rng(0)
    t_out = rng.uniform(-2, 12, n)
    rts = rng.uniform(15, 22, n)
    flow = 41.0 - 0.7 * t_out + 1.9 * (rts - 18.5)
    energy = 0.4 - 0.12 * t_out + 0.11 * rts
    return t_out, flow, rts, energy


def test_fit_recovers_the_generating_parameters():
    t_out, flow, rts, energy = make_data()
    fitted = fit_pump_params(t_out, flow, rts, energy)
    assert fitted.flow_slope == pytest.approx(-0.7, abs=1e-6)
    assert flow_target(fitted, 5.0, 18.5) == pytest.approx(41.0 - 0.7 * 5.0, abs=1e-6)
    assert fitted.energy_per_rts == pytest.approx(0.11, abs=1e-6)
    assert fitted.energy_per_outdoor == pytest.approx(-0.12, abs=1e-6)


def test_fit_refuses_data_without_variation_or_enough_samples():
    t_out, flow, rts, energy = make_data()
    with pytest.raises(ModelFitError):
        fit_pump_params(t_out[:10], flow[:10], rts[:10], energy[:10])
    with pytest.raises(ModelFitError):
        fit_pump_params(t_out, flow, np.full_like(rts, 18.0), energy)
    with pytest.raises(ModelFitError):
        fit_pump_params(np.full_like(t_out, 5.0), flow, rts, energy)



from core.theria.plan.pump_model import (  # noqa: E402
    aux_kwh_per_hour,
    fit_aux_per_flow_c,
    total_kwh_per_hour,
)

WITH_AUX = replace(PRIOR_PUMP_PARAMS, aux_onset_c=44.0, aux_per_flow_c=0.15)


def test_the_electric_heater_only_costs_energy_above_its_flow_onset():
    below = flow_target(WITH_AUX, 8.0, 18.0)
    assert below < 44.0 and float(aux_kwh_per_hour(WITH_AUX, 8.0, 18.0)) == 0.0
    cold = float(aux_kwh_per_hour(WITH_AUX, -8.0, 18.0))  # flow target about 46 C
    assert cold == pytest.approx(0.15 * (flow_target(WITH_AUX, -8.0, 18.0) - 44.0))


def test_total_energy_adds_the_heater_to_the_compressor():
    t, rts = -8.0, 18.0
    assert float(total_kwh_per_hour(WITH_AUX, t, rts)) == pytest.approx(
        float(energy_kwh_per_hour(WITH_AUX, t, rts))
        + float(aux_kwh_per_hour(WITH_AUX, t, rts))
    )
    assert float(total_kwh_per_hour(PRIOR_PUMP_PARAMS, t, rts)) == pytest.approx(
        float(energy_kwh_per_hour(PRIOR_PUMP_PARAMS, t, rts))
    )  # the prior has no heater term


def test_lowering_the_setpoint_in_the_cold_saves_heater_energy_too():
    cold = -8.0
    saved = float(total_kwh_per_hour(WITH_AUX, cold, 18.0)) - float(
        total_kwh_per_hour(WITH_AUX, cold, 17.0)
    )
    compressor_only = float(energy_kwh_per_hour(WITH_AUX, cold, 18.0)) - float(
        energy_kwh_per_hour(WITH_AUX, cold, 17.0)
    )
    assert saved > compressor_only


def test_the_heater_coefficient_is_recovered_from_hourly_data():
    flow = np.tile(np.array([30.0, 40.0, 43.0, 45.0, 47.0, 49.0, 51.0]), 10)
    aux = 0.2 * np.maximum(0.0, flow - 44.0)
    assert fit_aux_per_flow_c(flow, aux, onset_c=44.0) == pytest.approx(0.2)


def test_without_hours_above_the_onset_the_heater_fit_is_refused():
    with pytest.raises(ModelFitError):
        fit_aux_per_flow_c(np.full(50, 38.0), np.zeros(50), onset_c=44.0)


STEEP_BELOW_18 = ((15.0, -9.0), (16.0, -6.0), (17.0, -3.0), (18.0, 0.0), (20.0, 3.8))


def with_levels(levels=STEEP_BELOW_18):
    return replace(PRIOR_PUMP_PARAMS, rts_levels=levels)


def test_a_measured_setpoint_moves_the_flow_by_what_the_pump_did():
    p = with_levels()
    assert flow_target(p, 5.0, 15.0) - flow_target(p, 5.0, 18.0) == pytest.approx(-9.0)


def test_a_setpoint_between_two_measured_ones_lies_between_them():
    p = with_levels()
    assert flow_target(p, 5.0, 15.5) - flow_target(p, 5.0, 18.0) == pytest.approx(-7.5)


def test_outside_the_measured_setpoints_the_flow_follows_the_usual_step():
    p = with_levels()
    assert flow_target(p, 5.0, 14.0) - flow_target(p, 5.0, 15.0) == pytest.approx(-1.9)
    assert flow_target(p, 5.0, 21.0) - flow_target(p, 5.0, 20.0) == pytest.approx(1.9)


def test_measured_setpoints_work_on_the_planners_grid_of_hours_and_setpoints():
    p = with_levels()
    grid = flow_target(p, np.array([[0.0], [5.0]]), np.array([[15.0, 18.0]]))
    assert grid.shape == (2, 2)
    assert grid[1, 0] - grid[1, 1] == pytest.approx(-9.0)


def test_measured_setpoints_survive_a_json_file():
    p = with_levels()
    again = PumpParams(**json.loads(json.dumps(asdict(p))))
    assert again == p
    assert flow_target(again, 5.0, 15.0) == flow_target(p, 5.0, 15.0)


def history_with_levels(rts_values, n_each=20):
    rts = np.repeat(np.asarray(rts_values, dtype=float), n_each)
    t_out = np.linspace(0.0, 10.0, len(rts))
    return t_out, rts


def test_the_setpoint_levels_are_fitted_from_what_the_pump_asked_for():
    t_out, rts = history_with_levels([15.0, 16.0, 17.0, 18.0])
    flow = 40.0 - 0.7 * t_out + 3.0 * (rts - 18.0)
    levels = dict(fit_rts_levels(t_out, flow, rts, 40.0, -0.7))
    assert levels == pytest.approx({15.0: -9.0, 16.0: -6.0, 17.0: -3.0, 18.0: 0.0})


def test_a_setpoint_seen_too_seldom_is_left_out():
    t_out, rts = history_with_levels([17.0, 18.0])
    rare = np.array([15.0, 15.0])
    levels = fit_rts_levels(
        np.r_[t_out, 5.0, 5.0],
        np.r_[40.0 - 0.7 * t_out, 20.0, 20.0],
        np.r_[rts, rare],
        40.0,
        -0.7,
    )
    assert [rts_c for rts_c, _ in levels] == [17.0, 18.0]


def test_an_hour_when_the_setpoint_changed_is_not_a_measurement():
    t_out, rts = history_with_levels([17.0, 18.0])
    levels = fit_rts_levels(
        np.r_[t_out, np.full(20, 5.0)],
        np.r_[40.0 - 0.7 * t_out, np.full(20, 20.0)],
        np.r_[rts, np.full(20, 17.3)],
        40.0,
        -0.7,
    )
    assert [rts_c for rts_c, _ in levels] == [17.0, 18.0]


def test_a_lower_setpoint_never_gets_a_higher_flow_than_a_higher_one():
    t_out, rts = history_with_levels([16.0, 17.0, 18.0])
    noisy = 40.0 - 0.7 * t_out + np.select([rts == 16.0, rts == 17.0], [1.0, -2.0], 0.0)
    levels = dict(fit_rts_levels(t_out, noisy, rts, 40.0, -0.7))
    assert levels[16.0] <= levels[17.0] <= levels[18.0]
    assert levels[16.0] == pytest.approx(-2.0)
