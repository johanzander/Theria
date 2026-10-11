import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.pump_model import (
    PRIOR_PUMP_PARAMS,
    energy_kwh_per_hour,
    fit_pump_params,
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


from dataclasses import replace  # noqa: E402

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
