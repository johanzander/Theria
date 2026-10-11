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
