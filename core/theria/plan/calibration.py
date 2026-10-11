"""Fitting the pump and room models to history, with plausibility checks."""

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.plan.calibration_data import (
    CalibrationInputs,
    HistorySource,
    WeatherHistory,
    load_calibration_inputs,
)
from core.theria.plan.daily_room import (
    DailyData,
    DailyRoomFit,
    daily_means,
    fit_daily_room,
    room_params_from_fit,
)
from core.theria.plan.pump_model import (
    DEFAULT_FLOW_PER_RTS,
    PumpParams,
    energy_kwh_per_hour,
    fit_aux_per_flow_c,
    fit_energy_model,
    fit_flow_curve,
)
from core.theria.plan.room_model import RoomParams
from core.theria.plan.settings import PlanSettings

# The heat curve is changed by hand; only recent data describes it.
FLOW_WINDOW_DAYS = 240
FLOW_MAX_OUTDOOR_C = (
    12.0  # the range the plan works in; the curve bends in mild weather
)
MIN_FLOW_ROWS = 200
MIN_ENERGY_ROWS = 500
MIN_FLOW_SLOPE, MAX_FLOW_SLOPE = -1.5, -0.2  # C of flow per C outdoors
MIN_MEAN_KWH, MAX_MEAN_KWH = 0.2, 4.0  # plausible mean heating electricity per hour


@dataclass(frozen=True)
class PumpFit:
    params: PumpParams
    flow_rows: int
    flow_rmse_c: float
    energy_rows: int
    energy_rmse_kwh: float
    aux_rows: int = 0  # hours above the heater onset that the heater term was fitted on


@dataclass(frozen=True)
class RoomFit:
    """How the room follows the flow temperature: hourly planner parameters and the daily fit.

    The fit knows the *response* (gain, time constant), not the room's temperature: its
    predictions are no better than "no change" (see `response.cv_*`).
    """

    params: RoomParams
    response: DailyRoomFit


@dataclass(frozen=True)
class FittedModels:
    pump: PumpFit
    room: RoomFit | None  # None when the room fit was rejected; room_error says why
    room_error: str | None
    fitted_at: datetime
    window_start: datetime
    window_end: datetime


def fit_pump(inputs: CalibrationInputs) -> PumpFit:
    """Fit flow curve (recent heating hours) and energy model (all heating hours)."""
    heating = (
        (inputs.hot_water == 0.0)
        & np.isfinite(inputs.t_out_c)
        & np.isfinite(inputs.heating_setpoint_c)
        & np.isfinite(inputs.rts_c)
        & (inputs.heating_setpoint_c > 15.0)
    )
    cold_enough = np.nan_to_num(inputs.t_out_c, nan=99.0) <= FLOW_MAX_OUTDOOR_C
    recent = np.zeros(len(heating), dtype=bool)
    recent[-FLOW_WINDOW_DAYS * 24 :] = True
    flow_rows = heating & recent & cold_enough
    if int(flow_rows.sum()) < MIN_FLOW_ROWS:
        raise ModelFitError(
            f"flow curve needs {MIN_FLOW_ROWS} heating hours below {FLOW_MAX_OUTDOOR_C} C "
            f"in the last {FLOW_WINDOW_DAYS} days, found {int(flow_rows.sum())}"
        )
    intercept, slope, rts_ref = fit_flow_curve(
        inputs.t_out_c[flow_rows],
        inputs.heating_setpoint_c[flow_rows],
        inputs.rts_c[flow_rows],
    )
    if not MIN_FLOW_SLOPE <= slope <= MAX_FLOW_SLOPE:
        raise ModelFitError(f"implausible heat curve slope {slope:.2f} C/C")
    energy_rows = heating & np.isfinite(inputs.energy_kwh)
    if int(energy_rows.sum()) < MIN_ENERGY_ROWS:
        raise ModelFitError(
            f"energy model needs {MIN_ENERGY_ROWS} heating hours with energy, "
            f"found {int(energy_rows.sum())}"
        )
    e0, e_out, e_rts = fit_energy_model(
        inputs.t_out_c[energy_rows],
        inputs.rts_c[energy_rows],
        inputs.energy_kwh[energy_rows],
    )
    if e_rts <= 0.0 or e_out >= 0.0:
        raise ModelFitError(
            f"implausible energy model (per room setpoint {e_rts:.3f}, per outdoor {e_out:.3f})"
        )
    params = PumpParams(
        flow_intercept=intercept,
        flow_slope=slope,
        rts_ref=rts_ref,
        flow_per_rts=DEFAULT_FLOW_PER_RTS,
        energy_intercept=e0,
        energy_per_outdoor=e_out,
        energy_per_rts=e_rts,
    )
    aux_rows = (
        heating
        & np.isfinite(inputs.aux_kwh)
        & (inputs.heating_setpoint_c > params.aux_onset_c)
    )
    try:
        aux_per_flow_c = fit_aux_per_flow_c(
            inputs.heating_setpoint_c[aux_rows],
            inputs.aux_kwh[aux_rows],
            params.aux_onset_c,
        )
    except ModelFitError:
        # too few cold, high-flow hours to see the heater: leave the term out (zero)
        aux_per_flow_c, aux_rows = 0.0, np.zeros(len(heating), dtype=bool)
    params = replace(params, aux_per_flow_c=aux_per_flow_c)
    predicted = energy_kwh_per_hour(
        params, inputs.t_out_c[energy_rows], inputs.rts_c[energy_rows]
    )
    mean_kwh = float(np.mean(predicted))
    if not MIN_MEAN_KWH <= mean_kwh <= MAX_MEAN_KWH:
        raise ModelFitError(
            f"implausible mean heating electricity {mean_kwh:.2f} kWh/h"
        )
    flow_pred = (
        intercept
        + slope * inputs.t_out_c[flow_rows]
        + DEFAULT_FLOW_PER_RTS * (inputs.rts_c[flow_rows] - rts_ref)
    )
    return PumpFit(
        params=params,
        flow_rows=int(flow_rows.sum()),
        flow_rmse_c=float(
            np.sqrt(np.mean((flow_pred - inputs.heating_setpoint_c[flow_rows]) ** 2))
        ),
        aux_rows=int(aux_rows.sum()),
        energy_rows=int(energy_rows.sum()),
        energy_rmse_kwh=float(
            np.sqrt(np.mean((predicted - inputs.energy_kwh[energy_rows]) ** 2))
        ),
    )


def fit_room(inputs: CalibrationInputs) -> RoomFit:
    """Fit the room's response to the flow temperature on daily means over several winters."""
    hours = inputs.hours
    days = [h for h in hours if h.hour == 0]
    data = DailyData(
        days=days,
        room=daily_means(hours, inputs.room_reference_c),
        t_out=daily_means(hours, inputs.t_out_c),
        flow=daily_means(hours, inputs.heating_setpoint_c),
        sun=daily_means(hours, inputs.sun_wm2),
    )
    response = fit_daily_room(data)
    return RoomFit(params=room_params_from_fit(response), response=response)


class Calibrator:
    """Loads history and fits both models. Raises ModelFitError when a fit is not trustworthy."""

    def __init__(
        self,
        source: HistorySource,
        weather: WeatherHistory,
        settings: PlanSettings,
        days: int = 1460,
    ):
        self._source = source
        self._weather = weather
        self._settings = settings
        self._days = days

    def run(self, now: datetime) -> FittedModels:
        inputs = load_calibration_inputs(
            self._source, self._weather, self._settings, now, self._days
        )
        pump = fit_pump(inputs)
        try:
            room, room_error = fit_room(inputs), None
        except ModelFitError as e:  # the pump fit is still useful without a room model
            room, room_error = None, f"{type(e).__name__}: {e}"
        return FittedModels(
            pump=pump,
            room=room,
            room_error=room_error,
            fitted_at=now,
            window_start=inputs.hours[0],
            window_end=inputs.hours[-1],
        )
