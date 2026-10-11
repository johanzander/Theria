"""Heat pump model: flow target and heating electricity versus outdoor temperature and room setpoint."""

from dataclasses import dataclass

import numpy as np

from core.theria.exceptions import ModelFitError

Array = float | np.ndarray  # scalars and numpy arrays are both accepted

DEFAULT_FLOW_PER_RTS = 1.9  # measured median, 227 steps 2022-2025


@dataclass(frozen=True)
class PumpParams:
    """Parameters of the pump model.

    flow_target = flow_intercept + flow_slope * t_out + flow_per_rts * (rts - rts_ref)
    energy      = energy_intercept + energy_per_outdoor * t_out + energy_per_rts * rts
    """

    flow_intercept: float
    flow_slope: float
    rts_ref: float
    flow_per_rts: float
    energy_intercept: float
    energy_per_outdoor: float
    energy_per_rts: float


# Priors from the measured winter 2025/26 curve and the rough counter regression
# (documentation/ivt-heat-pump.md). Replaced by fit_pump_params once data is loaded.
PRIOR_PUMP_PARAMS = PumpParams(
    flow_intercept=40.8,
    flow_slope=-0.68,
    rts_ref=18.0,
    flow_per_rts=DEFAULT_FLOW_PER_RTS,
    energy_intercept=0.31,
    energy_per_outdoor=-0.15,
    energy_per_rts=0.13,
)


def flow_target(params: PumpParams, t_out_c: Array, rts_c: Array) -> Array:
    """Pump flow temperature target (C). Works on scalars and numpy arrays."""
    return (
        params.flow_intercept
        + params.flow_slope * t_out_c
        + params.flow_per_rts * (rts_c - params.rts_ref)
    )


def energy_kwh_per_hour(params: PumpParams, t_out_c: Array, rts_c: Array) -> Array:
    """Heating electricity (kWh per hour), never negative."""
    raw = (
        params.energy_intercept
        + params.energy_per_outdoor * t_out_c
        + params.energy_per_rts * rts_c
    )
    return np.maximum(raw, 0.0)


def fit_flow_curve(
    t_out_c: np.ndarray,
    flow_target_c: np.ndarray,
    rts_c: np.ndarray,
    flow_per_rts: float = DEFAULT_FLOW_PER_RTS,
) -> tuple[float, float, float]:
    """Fit the heat curve; returns (flow_intercept, flow_slope, rts_ref).

    The room setpoint does not need to vary: its effect is the fixed `flow_per_rts`.
    """
    if len(t_out_c) < 24 or not (len(t_out_c) == len(flow_target_c) == len(rts_c)):
        raise ModelFitError(
            "flow curve fit needs equally long arrays, at least 24 samples"
        )
    if np.ptp(t_out_c) < 3.0:
        raise ModelFitError(
            "flow curve fit needs at least 3 C spread in outdoor temperature"
        )
    rts_ref = float(np.mean(rts_c))
    adjusted = flow_target_c - flow_per_rts * (rts_c - rts_ref)
    slope, intercept = np.polyfit(t_out_c, adjusted, 1)
    return float(intercept), float(slope), rts_ref


def fit_energy_model(
    t_out_c: np.ndarray, rts_c: np.ndarray, energy_kwh_per_h: np.ndarray
) -> tuple[float, float, float]:
    """Fit heating electricity; returns (intercept, per outdoor C, per room setpoint C)."""
    n = len(t_out_c)
    if n < 24 or not (n == len(rts_c) == len(energy_kwh_per_h)):
        raise ModelFitError("energy fit needs equally long arrays, at least 24 samples")
    if np.ptp(t_out_c) < 3.0:
        raise ModelFitError(
            "energy fit needs at least 3 C spread in outdoor temperature"
        )
    if np.ptp(rts_c) < 1.0:
        raise ModelFitError("energy fit needs at least 1 C spread in room setpoint")
    design = np.column_stack([np.ones(n), t_out_c, rts_c])
    coef, *_ = np.linalg.lstsq(design, energy_kwh_per_h, rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def fit_pump_params(
    t_out_c: np.ndarray,
    flow_target_c: np.ndarray,
    rts_c: np.ndarray,
    energy_kwh_per_h: np.ndarray,
    flow_per_rts: float = DEFAULT_FLOW_PER_RTS,
) -> PumpParams:
    """Fit both parts on the same hourly heating-only samples."""
    intercept, slope, rts_ref = fit_flow_curve(
        t_out_c, flow_target_c, rts_c, flow_per_rts
    )
    e0, e_out, e_rts = fit_energy_model(t_out_c, rts_c, energy_kwh_per_h)
    return PumpParams(
        flow_intercept=intercept,
        flow_slope=slope,
        rts_ref=rts_ref,
        flow_per_rts=flow_per_rts,
        energy_intercept=e0,
        energy_per_outdoor=e_out,
        energy_per_rts=e_rts,
    )
