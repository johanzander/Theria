"""Plan service: reads live data, computes the plan and keeps the latest snapshots.

Shadow mode never writes to Home Assistant. Active mode (explicitly confirmed) lets the executor
write the room setpoint and reports trouble (the electric heater running while it is raised).
"""

import asyncio
import logging
import os
import re
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

import numpy as np

from core.theria.exceptions import (
    ConfigurationError,
    PriceDataError,
    SensorError,
    SetpointRestoreError,
    TheriaError,
    WeatherDataError,
)
from core.theria.ha_client import HAClient
from core.theria.plan.calibration import Calibrator, FittedModels
from core.theria.plan.calibration_data import FLOW_PLAUSIBLE_C, OUTDOOR_PLAUSIBLE_C
from core.theria.plan.data_health import (
    COUNTED_WINDOW_HOURS,
    WINDOW_HOURS,
    counted_hours_check,
    history_checks,
    live_h60_checks,
    price_feed_check,
)
from core.theria.plan.day_view import build_day_view, day_hours
from core.theria.plan.debug_export import (
    build_report,
    key_findings,
    read_log_lines,
)
from core.theria.plan.disturbance import DisturbanceEstimator, Sample
from core.theria.plan.energy_forecast import trailing_means
from core.theria.plan.event_log import EventLog
from core.theria.plan.exec_trace import ExecTraceStore
from core.theria.plan.executor import MAX_PLAN_AGE, ExecutionResult, PlanExecutor
from core.theria.plan.heat_shift import plan_shift
from core.theria.plan.lk_executor import LKExecutor, LKResult
from core.theria.plan.model_catalog import build_catalog
from core.theria.plan.morning import MorningRules, update_margin
from core.theria.plan.night_drift import NightDrift
from core.theria.plan.outcomes import DayOutcome, OutcomeInputs
from core.theria.plan.parameter_log import ParameterLog
from core.theria.plan.plan_model import (
    HourPlan,
    Plan,
    PlanInputs,
    sun_filtered_sequence,
)
from core.theria.plan.plan_record import PlanRecord, PlanRecordStore
from core.theria.plan.plan_run import PlanRunStore, make_run
from core.theria.plan.preflight import Check, preflight_checks
from core.theria.plan.prices import PricePeriod, hourly_prices
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, PumpParams
from core.theria.plan.room_level import room_level_c
from core.theria.plan.room_model import (
    PRIOR_ROOM_PARAMS,
    RoomParams,
    RoomState,
    day_heat_c_per_h,
    with_measured_response,
)
from core.theria.plan.scoring import LeadScore, ScoreBoard
from core.theria.plan.settings import SUPPORTED_MODES, PlanSettings
from core.theria.plan.settings_manager import SettingsManager
from core.theria.plan.shift_rules import (
    NORMAL,
    OPENING,
    SalonStatus,
    ShiftReadings,
    ShiftWindow,
    thermostat_target,
)
from core.theria.plan.skill import SkillTracker
from core.theria.plan.snapshot import ComfortOutlook, PlanSnapshot
from core.theria.plan.state_store import PlanState, PlanStateStore
from core.theria.plan.trim import trim_night, trim_room, trim_warmup
from core.theria.plan.tuning import Recommendation, build_recommendations
from core.theria.plan.warmup import WarmupFit
from core.theria.plan.weather import OpenMeteoClient, WeatherHour
from core.theria.plan.zone_heat import ZoneMeasured
from core.theria.sensor_guard import SensorGuard

logger = logging.getLogger(__name__)

SLAB_INIT_FRACTION = 0.5  # rough prior: slab sits half way between room and flow target
SUN_WARMUP_HOURS = 24
LOG_KEPT = 200
ENTITY_ID = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")
TUNING_DAYS = 14
TUNING_CACHE = timedelta(
    hours=1
)  # the measured days change slowly; the query is not cheap
LAG_HOURS = 72  # the energy model's trailing outdoor means
ENERGY_SETPOINTS_C = (
    12.0,
    24.0,
)  # the energy table covers every setpoint the plan may use
SNAPSHOTS_KEPT = 96 * 2  # two days at one snapshot per 15 minutes


class Safety(Protocol):
    def check(self, raised: bool = False, deviated: bool = True) -> str | None: ...

    def use_settings(self, settings: PlanSettings) -> None: ...


class Notifier(Protocol):
    def notify(self, notification_id: str, title: str, message: str) -> None: ...

    def dismiss(self, notification_id: str) -> None: ...


class ForecastCheck(Protocol):
    def run_due(self) -> int: ...


class HeatingTable(Protocol):
    def table(
        self,
        t_out_c: np.ndarray,
        t24_c: np.ndarray,
        t72_c: np.ndarray,
        sun_wm2: np.ndarray,
        wind_kmh: np.ndarray,
        setpoints_c: np.ndarray,
    ) -> np.ndarray: ...


class EnergyForecast(Protocol):
    def run_due(self) -> int: ...

    def heating_model(self) -> HeatingTable | None: ...


OutcomeLoader = Callable[[datetime, datetime], list[DayOutcome]]
ActualsLoader = Callable[[datetime, datetime], OutcomeInputs]
ZoneLoader = Callable[[datetime, datetime], ZoneMeasured]
TROUBLE_NOTIFICATION = "theria_plan_trouble"
DAILY_NOTIFICATION = "theria_plan_daily"
MARGIN_NOTIFICATION = "theria_plan_margin"
NIGHT_MODEL_NOTIFICATION = "theria_plan_night_model"
NIGHT_ALERT_FACTOR = (
    1.8  # recent night errors this much above the fit error are a warning
)
NIGHT_ALERT_MIN_C = 0.4  # ... and at least this large
OUTCOME_LOOKBACK_DAYS = 30


def _floor_hour(moment: datetime) -> datetime:
    return moment.replace(minute=0, second=0, microsecond=0)


def _strings(value: Any) -> list[str]:
    """Every string inside nested dicts, lists and tuples."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in [*value.keys(), *value.values()] for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


class PlanService:
    """Computes the price-driven plan. Shadow mode: never writes to Home Assistant."""

    def __init__(
        self,
        ha_client: HAClient,
        settings: PlanSettings,
        weather: OpenMeteoClient,
        price_periods: Callable[[], list[PricePeriod]],
        pump_params: PumpParams = PRIOR_PUMP_PARAMS,
        room_params: RoomParams = PRIOR_ROOM_PARAMS,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        interval_seconds: int = 900,
        calibrator: Calibrator | None = None,
        state_store: PlanStateStore | None = None,
        refit_interval_hours: float = 24.0,
        retry_interval_hours: float = 6.0,
        sensor_guard: SensorGuard | None = None,
        executor: PlanExecutor | None = None,
        lk_executor: LKExecutor | None = None,
        plan_record: PlanRecordStore | None = None,
        plan_runs: PlanRunStore | None = None,
        exec_trace: ExecTraceStore | None = None,
        log_dir: str | None = None,
        settings_manager: SettingsManager | None = None,
        safety: Safety | None = None,
        notifier: Notifier | None = None,
        outcome_loader: OutcomeLoader | None = None,
        parameter_log: ParameterLog | None = None,
        event_log: EventLog | None = None,
        actuals_loader: ActualsLoader | None = None,
        price_history: Callable[[datetime, datetime], list[PricePeriod]] | None = None,
        forecast_check: ForecastCheck | None = None,
        energy_forecast: EnergyForecast | None = None,
        zone_loader: ZoneLoader | None = None,
    ):
        self._ha = ha_client
        self._settings = settings
        self._weather = weather
        self._pump = pump_params
        self._room = with_measured_response(room_params)
        self._now = now
        self._interval_seconds = interval_seconds
        self._wake = (
            asyncio.Event()
        )  # set to run the next cycle now instead of after the interval
        self._mode = settings.mode
        self._snapshots: deque[PlanSnapshot] = deque(maxlen=SNAPSHOTS_KEPT)
        self._task: asyncio.Task | None = None
        self._calibrator = calibrator
        self._sensor_guard = sensor_guard
        self._state_store = state_store
        self._refit_interval = timedelta(hours=refit_interval_hours)
        self._retry_interval = timedelta(hours=retry_interval_hours)
        self._models: FittedModels | None = None
        self._last_fit_error: str | None = None
        self._night: NightDrift | None = None
        self._morning_margin_c = 0.0
        self._night_alert_active = False
        self._margin_changed_on: date | None = None
        self._morning_rules = MorningRules()
        self._night_error: str | None = None
        self._warmup: WarmupFit | None = None
        self._warmup_error: str | None = None
        self._next_refit_at: datetime | None = None
        self._scoreboard = ScoreBoard()
        self._parameter_log = parameter_log
        self._executor = executor
        self._lk = lk_executor
        self._plan_record = plan_record
        self._plan_runs = plan_runs
        self._exec_trace = exec_trace
        self._log_dir = log_dir
        self._settings_manager = settings_manager
        self._safety = safety
        self._notifier = notifier
        self._outcome_loader = outcome_loader
        self._price_periods = price_periods
        self._price_history = price_history or self._periods_between
        self._reviewed_on: date | None = None
        self._plan_error: str | None = None
        self._last_trouble: str | None = None
        # set_mode and the control cycle never interleave
        self._lock = threading.RLock()
        self._event_log = event_log
        self._disturbance = DisturbanceEstimator()
        self._skill = SkillTracker()
        self._authority_day: tuple[date, float] | None = None
        self._shift_window: ShiftWindow | None = None
        self._salon_status: Callable[[], SalonStatus] | None = None
        self._tuning_days: tuple[datetime, list[DayOutcome]] | None = None
        self._actuals_loader = actuals_loader
        self._forecast_check = forecast_check
        self._energy_forecast = energy_forecast
        self._zone_loader = zone_loader
        self._restore_failed = (
            False  # leaving active mode failed: no window may be reapplied
        )
        self._events: deque[dict[str, Any]] = deque(
            [] if event_log is None else event_log.read(LOG_KEPT), maxlen=LOG_KEPT
        )
        if state_store is not None:
            state = state_store.load()
            if state.mode is not None:
                self._mode = state.mode
            if state.models is not None:
                self._apply_models(state.models)
                self._next_refit_at = state.models.fitted_at + self._refit_interval
            self._morning_margin_c = state.morning_margin_c
            self._margin_changed_on = state.margin_changed_on
            self._reviewed_on = state.reviewed_on
            self._last_trouble = state.last_trouble
            if state.executor is not None and executor is not None:
                executor.load_state(state.executor)
            if state.lk is not None and lk_executor is not None:
                lk_executor.load_state(state.lk)
            if state.skill is not None:
                self._skill.load_state(state.skill)
            if state.shift_window is not None:
                self._shift_window = ShiftWindow.from_dict(state.shift_window)

    def now(self) -> datetime:
        return self._now()

    def _read_float(
        self, entity_id: str, bounds: tuple[float, float] | None = None
    ) -> float:
        """Numeric state of an entity; raises SensorError for unavailable/unknown, and for a value
        outside `bounds` (the H60 answers with garbage now and then)."""
        raw = self._ha.get_state(entity_id)["state"]
        try:
            value = float(raw)
        except (TypeError, ValueError) as e:
            raise SensorError(f"{entity_id} has non-numeric state {raw!r}") from e
        if bounds is not None and not bounds[0] <= value <= bounds[1]:
            raise SensorError(f"{entity_id} reads {value}, outside {bounds}")
        return value

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str, confirm: bool = False) -> None:
        """Change the mode; it is persisted so that a restart keeps it.

        Active mode writes to the heat pump, so it needs `confirm`, and it is refused while
        the old setpoint automation is on. Leaving it first puts the base setpoint back; if
        that fails the mode stays active (and the error says so) so that control can retry.
        """
        if mode not in SUPPORTED_MODES:
            raise ConfigurationError(
                f"mode must be one of {SUPPORTED_MODES}, got {mode!r}"
            )
        with self._lock:
            was_active = self._mode == "active"
            if mode == "active":
                if self._executor is None:
                    raise ConfigurationError(
                        "active mode is not available: no executor"
                    )
                if not confirm:
                    raise ConfigurationError(
                        "active mode writes to the heat pump and needs confirm=true"
                    )
                if not was_active:
                    self._refuse_while_automation_is_on()
            if was_active and mode != "active" and self._executor is not None:
                failure = self._restore_all()
                self._restore_failed = failure is not None
                if failure is not None:
                    raise SetpointRestoreError(
                        f"the base setpoints could not be restored ({failure}); "
                        "the mode stays active"
                    )
            if mode == "active" and not was_active:
                self._reviewed_on = None
            if mode != self._mode:
                self._log("mode", mode, f"from {self._mode}")
            self._mode = mode
            self._save_state()

    def parameters(self) -> dict[str, Any]:
        """Everything that is adjusted over time, as plain numbers."""
        pump = self._pump
        params: dict[str, Any] = {
            "flow_intercept": pump.flow_intercept,
            "flow_slope": pump.flow_slope,
            "flow_per_rts": pump.flow_per_rts,
            "energy_intercept": pump.energy_intercept,
            "energy_per_outdoor": pump.energy_per_outdoor,
            "energy_per_rts": pump.energy_per_rts,
            "aux_per_flow_c": pump.aux_per_flow_c,
            "morning_margin_c": self._morning_margin_c,
            "disturbance_c_per_h": self._disturbance.at(self._now()),
            "authority": self._authority(),
            "sun_swing_c_per_wm2": self._room.sun_swing_c_per_wm2,
            "sun_lag_h": self._room.sun_lag_h,
        }
        fitted = self._models
        if fitted is not None:
            params["flow_rmse_c"] = fitted.pump.flow_rmse_c
            params["energy_rmse_kwh"] = fitted.pump.energy_rmse_kwh
            if fitted.room is not None:
                response = fitted.room.response
                params["room_gain_c_per_c"] = response.gain_c_per_c
                params["room_rate_per_day"] = response.rate_per_day
                params["room_time_constant_days"] = response.time_constant_days
        if self._warmup is not None:
            params["warmup_scale"] = self._warmup.scale
            params["warmup_applied_scale"] = self._warmup.applied_scale
            params["warmup_response_c_per_c"] = self._warmup.response_c_per_c
            params["cooling_response_c_per_c"] = self._warmup.cooling_response_c_per_c
            params["cooling_scale"] = self._warmup.cooling_scale
        if self._night is not None:
            params["night_intercept_c"] = self._night.intercept_c
            params["night_slope_c_per_c"] = self._night.slope_c_per_c
            params["night_level_c_per_c"] = self._night.level_c_per_c
            params["night_nights"] = self._night.nights
            params["night_rmse_c"] = self._night.rmse_c
        return params

    def parameter_history(self, limit: int) -> list[dict[str, Any]]:
        return [] if self._parameter_log is None else self._parameter_log.read(limit)

    def _record_parameters(self, kind: str, note: str) -> None:
        if self._parameter_log is None:
            return
        try:
            self._parameter_log.append(self._now(), kind, note, self.parameters())
        except OSError:
            logger.exception("could not write the parameter history")

    def _check_night_model(self) -> None:
        """Warn when the night model misses the recent nights by much more than it should."""
        night = self._night
        if night is None or not night.recent:
            return
        errors = np.array(
            [o.measured_change_c - o.predicted_change_c for o in night.recent]
        )
        recent_rmse = float(np.sqrt(np.mean(errors**2)))
        bad = recent_rmse > max(NIGHT_ALERT_MIN_C, NIGHT_ALERT_FACTOR * night.rmse_c)
        if bad and not self._night_alert_active:
            message = (
                f"The night cooling model missed the last {len(night.recent)} nights by "
                f"{recent_rmse:.2f} C (rms), against {night.rmse_c:.2f} C when it was fitted. "
                "Something in the house or the sensors has changed."
            )
            self._log("model", "night model off", message)
            self._notify(NIGHT_MODEL_NOTIFICATION, "Theria: night model off", message)
        if not bad and self._night_alert_active and self._notifier is not None:
            try:
                self._notifier.dismiss(NIGHT_MODEL_NOTIFICATION)
            except Exception:  # a failed dismissal must not stop the refit
                logger.exception("could not dismiss the night model notification")
        self._night_alert_active = bad

    @staticmethod
    def _comfort_outlook(
        result: Any,
        settings: PlanSettings,
        room: RoomParams,
    ) -> ComfortOutlook | None:
        hours = result.hours
        for k in range(len(hours) - 1):
            if settings.in_night(hours[k].start) and not settings.in_night(
                hours[k + 1].start
            ):
                hour = hours[k]
                expected = hour.room_c_end + hour.sun_swing_c
                deficit = max(0.0, settings.target_c - expected)
                cap = settings.floor_mix_max_c
                extra = room.k_slab * max(0.0, cap - hour.flow_target_c)
                lead = (
                    0.0 if deficit == 0.0 else (deficit / extra if extra > 0 else None)
                )
                return ComfortOutlook(
                    hour.start + timedelta(hours=1),
                    expected,
                    settings.target_c,
                    deficit,
                    extra,
                    lead,
                )
        return None

    def _warmed_room(self) -> RoomParams:
        """The room model with its warming speed corrected by what has been measured."""
        if self._warmup is None:
            return self._room
        return replace(
            self._room, k_slab=self._room.k_slab * self._warmup.applied_scale
        )

    def _authority(self) -> float:
        """The share of the setpoint range the plan may use, fixed for the local day: it changes the
        action set, and a range that moved between replans would move the plan with it.
        """
        day = self._now().astimezone(ZoneInfo(self._settings.timezone)).date()
        if self._authority_day is None or self._authority_day[0] != day:
            self._authority_day = (
                day,
                self._skill.authority(self._now(), self._settings.min_authority),
            )
        return self._authority_day[1]

    def _effective_settings(self) -> PlanSettings:
        """The settings the plan runs on: the night kept warmer by the learned margin, and the
        ceiling lowered by the rise the floor adds after the valves close.
        """
        s = self._settings
        # the swing of the setpoint around its base grows with the model's measured skill
        authority = self._authority()
        step = s.rts_step_c
        down = round((s.rts_base_c - s.rts_min_c) * authority / step) * step
        up = round((s.rts_max_c - s.rts_base_c) * authority / step) * step
        s = replace(s, rts_min_c=s.rts_base_c - down, rts_max_c=s.rts_base_c + up)
        if self._morning_margin_c <= 0.0 or s.night_start_hour == s.night_end_hour:
            return s
        return replace(
            s,
            night_target_c=min(s.target_c, s.night_target_c + self._morning_margin_c),
            night_floor_c=min(s.floor_c, s.night_floor_c + self._morning_margin_c),
        )

    def _learn_morning_margin(self, outcomes: list[DayOutcome], today: date) -> None:
        s = self._settings
        if s.night_start_hour == s.night_end_hour:
            return
        decision = update_margin(
            self._morning_margin_c,
            outcomes,
            s.target_c,
            today,
            self._margin_changed_on or date.min,
            self._morning_rules,
            cap_c=s.target_c - s.night_target_c,
        )
        if decision.action == "hold":
            return
        self._morning_margin_c = decision.margin_c
        self._margin_changed_on = decision.changed_on
        self._log("margin", decision.action, decision.reason)
        self._record_parameters("margin", decision.reason)
        self._notify(MARGIN_NOTIFICATION, "Theria: night margin", decision.reason)

    def _log(self, kind: str, action: str, detail: str) -> None:
        event = {"at": self._now(), "kind": kind, "action": action, "detail": detail}
        self._events.append(event)
        if self._event_log is not None:
            self._event_log.append(event)

    def action_log(self) -> list[dict[str, Any]]:
        """What the plan did (oldest first): setpoint writes, trouble and mode changes."""
        return list(self._events)

    def preflight(self) -> list[Check]:
        """What to read before activating: the state of everything active mode depends on."""
        fitted = self._models
        now = self._now()
        return [
            *preflight_checks(
                self._ha,
                self._settings,
                self.latest(),
                pump_fitted=fitted is not None,
                room_fitted=fitted is not None and fitted.room is not None,
                now=now,
            ),
            *self._data_checks(now),
        ]

    def _data_checks(self, now: datetime) -> list[Check]:
        """Prices, stored history and the H60's live readings: see data_health."""
        s = self._settings
        try:
            checks = [price_feed_check(self._price_periods(), now, s.timezone)]
        except TheriaError as e:
            checks = [Check("price feed", "fail", f"cannot read the prices: {e}")]
        checks += live_h60_checks(self._ha, s)
        if self._actuals_loader is None:
            return checks
        stop = _floor_hour(now)
        try:
            inputs = self._actuals_loader(
                stop - timedelta(hours=max(WINDOW_HOURS, COUNTED_WINDOW_HOURS)), stop
            )
        except TheriaError as e:
            return [
                *checks,
                Check("history", "warn", f"cannot read the stored history: {e}"),
            ]
        return [*checks, *history_checks(inputs, s), counted_hours_check(inputs, s)]

    def _restore_all(self) -> str | None:
        """Pump setpoint and LK thermostats back to their base; a failure reason or None."""
        assert self._executor is not None
        failures = []
        pump = self._executor.restore_base()
        if pump.action == "failed":
            failures.append(f"pump: {pump.reason}")
        if self._lk is not None:
            lk = self._lk.restore_base()
            if lk.action == "failed":
                failures.append(f"LK: {lk.reason}")
        return "; ".join(failures) or None

    def _automation_is_on(self) -> bool:
        state = self._ha.get_state(self._settings.conflicting_automation_entity)
        return str(state["state"]) == "on"

    def _refuse_while_automation_is_on(self) -> None:
        entity = self._settings.conflicting_automation_entity
        try:
            conflict = self._automation_is_on()
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            raise ConfigurationError(f"cannot check {entity}: {e}") from e
        if conflict:
            raise ConfigurationError(
                f"{entity} is on and would fight the plan; turn it off first"
            )

    def _save_state(self) -> None:
        if self._state_store is not None:
            self._state_store.save(
                PlanState(
                    mode=self._mode,
                    models=self._models,
                    reviewed_on=self._reviewed_on,
                    morning_margin_c=self._morning_margin_c,
                    margin_changed_on=self._margin_changed_on,
                    last_trouble=self._last_trouble,
                    executor=(
                        None
                        if self._executor is None
                        else self._executor.export_state()
                    ),
                    lk=None if self._lk is None else self._lk.export_state(),
                    skill=self._skill.export_state(),
                    shift_window=(
                        None
                        if self._shift_window is None
                        else self._shift_window.to_dict()
                    ),
                )
            )

    @property
    def pump_params(self) -> PumpParams:
        return self._pump

    def savings(self, days: int) -> dict[str, Any]:
        """Measured cost against the counterfactual baseline, per day, for the last days."""
        if self._outcome_loader is None:
            raise ConfigurationError("savings are not available: no history source")
        now = self._now()
        stop = _floor_hour(now)
        start = datetime.combine(
            now.date() - timedelta(days=days - 1), datetime.min.time(), UTC
        )
        outcomes = self._outcome_loader(start, stop)
        return {
            "from": start,
            "to": stop,
            "days": outcomes,
            "total_saving_sek": float(sum(d.saving_sek for d in outcomes)),
            "total_actual_cost_sek": float(sum(d.actual_cost_sek for d in outcomes)),
            "total_baseline_cost_sek": float(
                sum(d.baseline_cost_sek for d in outcomes)
            ),
        }

    def settings_catalog(self) -> list[dict[str, Any]]:
        """The settings that can be changed in the UI, with value, default and where the value comes from."""
        return (
            [] if self._settings_manager is None else self._settings_manager.catalog()
        )

    def update_settings(self, values: dict[str, Any]) -> list[dict[str, Any]]:
        """Check, save and apply a change at once; raises ConfigurationError and changes nothing if invalid."""
        if self._settings_manager is None:
            raise ConfigurationError("settings cannot be changed here")
        with self._lock:
            self._use_settings(self._settings_manager.update(values))
        return self.settings_catalog()

    def reset_setting(self, key: str) -> list[dict[str, Any]]:
        if self._settings_manager is None:
            raise ConfigurationError("settings cannot be changed here")
        with self._lock:
            self._use_settings(self._settings_manager.reset(key))
        return self.settings_catalog()

    def _use_settings(self, settings: PlanSettings) -> None:
        self._settings = settings
        if self._executor is not None:
            self._executor.use_settings(settings)
        if self._lk is not None:
            self._lk.use_settings(settings)
        if self._safety is not None:
            self._safety.use_settings(settings)
        self._save_state()

    def model_catalog(self) -> list[dict[str, Any]]:
        """Every parameter and weight the plan runs on, with what it is and where it comes from."""
        now = self._now()
        latest = self.latest()
        return build_catalog(
            settings=self._effective_settings(),
            pump=self._pump,
            pump_fitted=self._models is not None,
            room=self._room,
            room_fitted=self._models is not None and self._models.room is not None,
            night=self._night,
            warmup=self._warmup,
            morning_margin_c=self._morning_margin_c,
            disturbance_c_per_h=self._disturbance.at(now),
            authority=self._authority(),
            skill_error_c_per_h=self._skill.error(now),
            plan=None if latest is None else latest.plan,
        )

    def execution_status(self) -> dict[str, Any]:
        result: ExecutionResult | None = (
            None if self._executor is None else self._executor.last_result
        )
        lk_result: LKResult | None = None if self._lk is None else self._lk.last_result
        return {
            "mode": self._mode,
            "reviewed_on": self._reviewed_on,
            "plan_error": self._plan_error,
            "last_execution": result,
            "last_lk": lk_result,
            "authority": self._authority(),
            "morning_margin_c": self._morning_margin_c,
            "last_trouble": self._last_trouble,
        }

    def _apply_models(self, models: FittedModels) -> None:
        self._models = models
        self._night = models.night_drift
        self._night_error = models.night_drift_error
        self._warmup = models.warmup
        self._warmup_error = models.warmup_error
        self._pump = models.pump.params
        if models.room is not None:
            self._room = with_measured_response(models.room.params)

    def refit(self) -> FittedModels:
        """Fit both models to history now; keeps the old models if the fit is rejected."""
        if self._calibrator is None:
            raise ConfigurationError("calibration is not configured")
        try:
            models = self._calibrator.run(self._now())
        except TheriaError as e:
            self._schedule_refit_retry(e)
            raise
        now = self._now()
        previous = self._models
        if models.room is None and previous is not None and previous.room is not None:
            # a rejected room fit must not discard the one that was accepted before
            models = replace(models, room=previous.room)
        if (
            models.night_drift is None
            and previous is not None
            and previous.night_drift is not None
        ):
            models = replace(models, night_drift=previous.night_drift)
        if (
            models.warmup is None
            and previous is not None
            and previous.warmup is not None
        ):
            models = replace(models, warmup=previous.warmup)
        notes: list[str] = []
        if previous is not None:
            night, night_notes = trim_night(previous.night_drift, models.night_drift)
            warmup, warmup_notes = trim_warmup(previous.warmup, models.warmup)
            room, room_notes = trim_room(previous.room, models.room)
            models = replace(models, night_drift=night, warmup=warmup, room=room)
            notes = room_notes + night_notes + warmup_notes
        self._apply_models(models)
        self._record_parameters(
            "refit", "; ".join(["nightly refit of the models", *notes])
        )
        self._check_night_model()
        self._last_fit_error = None
        self._next_refit_at = now + self._refit_interval
        self._save_state()
        return models

    def _schedule_refit_retry(self, error: Exception) -> None:
        self._last_fit_error = f"{type(error).__name__}: {error}"
        self._next_refit_at = self._now() + self._retry_interval

    def refit_due(self) -> bool:
        return self._calibrator is not None and (
            self._next_refit_at is None or self._now() >= self._next_refit_at
        )

    def latest(self) -> PlanSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None

    def _day_heat(self, moment: datetime) -> float:
        return day_heat_c_per_h(
            moment.astimezone(ZoneInfo(self._settings.timezone)).hour
        )

    def _extra_drift(self, starts: list[datetime]) -> np.ndarray:
        """What the room does on top of the model: the measured day heat. The disturbance is measured
        but not carried forward: in the backtest of 27 Oct-4 Nov it made the 6-12 h forecast worse.
        """
        return np.array([self._day_heat(start) for start in starts])

    def _prices_today(
        self, periods: list[PricePeriod], first_hour: datetime
    ) -> np.ndarray:
        """The hourly prices from local midnight up to `first_hour`, as far back as they are known."""
        tz = ZoneInfo(self._settings.timezone)
        midnight = first_hour.astimezone(tz).replace(hour=0, minute=0, second=0)
        prices: list[float] = []
        hour = first_hour
        while hour > midnight:
            hour -= timedelta(hours=1)
            try:
                prices.append(hourly_prices(periods, hour)[0][1])
            except PriceDataError:
                break
        return np.array(prices[::-1], dtype=float)

    def _sample_disturbance(
        self, now: datetime, room_c: float, weather: dict[datetime, WeatherHour]
    ) -> None:
        """Give the disturbance estimator what was measured this cycle."""
        s = self._settings
        try:
            sample = Sample(
                at=now,
                room_c=room_c,
                flow_c=min(
                    self._read_float(s.flow_measured_entity, FLOW_PLAUSIBLE_C),
                    s.floor_mix_max_c,
                ),
                valve=self._open_fraction(),
                t_out_c=self._read_float(s.outdoor_entity, OUTDOOR_PLAUSIBLE_C),
                sun_wm2=self._forecast_at(weather, _floor_hour(now)).shortwave_wm2,
            )
        except (RuntimeError, ValueError, KeyError, TypeError, WeatherDataError):
            logger.debug(
                "no complete sample for the disturbance estimator", exc_info=True
            )
            return
        innovation = self._disturbance.update(
            sample,
            replace(self._warmed_room(), drift_c_per_h=self._day_heat(now)),
            s.lk_base_c,
        )
        if innovation is not None:
            self._skill.add(innovation)

    def _energy_table(
        self, starts: list[datetime], weather: dict[datetime, WeatherHour], bias: float
    ) -> dict[str, np.ndarray]:
        """The energy forecast's heating per hour and setpoint, with the outdoor history it needs for
        its trailing means; nothing until its model has been fitted."""
        model = (
            None
            if self._energy_forecast is None
            else self._energy_forecast.heating_model()
        )
        if model is None:
            return {}
        hours = [starts[0] - timedelta(hours=k) for k in range(LAG_HOURS, 0, -1)]
        hours += starts
        t_out = np.array(
            [weather[h].temperature_c + bias if h in weather else np.nan for h in hours]
        )
        t24, t72 = trailing_means(t_out)
        future = slice(LAG_HOURS, None)
        setpoints = np.arange(
            ENERGY_SETPOINTS_C[0],
            ENERGY_SETPOINTS_C[1] + 1e-9,
            self._settings.rts_step_c,
        )
        table = model.table(
            t_out[future],
            t24[future],
            t72[future],
            np.array(
                [weather[h].shortwave_wm2 if h in weather else 0.0 for h in starts]
            ),
            np.array([weather[h].wind_kmh if h in weather else 0.0 for h in starts]),
            setpoints,
        )
        return {"energy_setpoints_c": setpoints, "energy_kwh_table": table}

    def use_salon_status(self, status: Callable[[], SalonStatus]) -> None:
        """Where the heat shift reads the salon from; without it the salon is not looked after."""
        self._salon_status = status

    def _salon_reading(self) -> SalonStatus | None:
        if self._salon_status is None:
            return None
        try:
            return self._salon_status()
        except (TheriaError, RuntimeError, ValueError, KeyError, TypeError):
            logger.warning("cannot read the salon", exc_info=True)
            return SalonStatus(None, 0.0)

    def _shift_readings(self, room_c: float) -> ShiftReadings | None:
        rooms = [
            r
            for r in (self._room_c(e) for e in self._settings.room_thermostat_entities)
            if r is not None
        ]
        offsets = tuple(r - float(np.mean(rooms)) for r in rooms)
        return ShiftReadings(room_c, self._salon_reading(), self._shift_window, offsets)

    def _keep_shift_window(self, window: ShiftWindow | None) -> None:
        if window == self._shift_window:
            return
        if window is not None and (
            self._shift_window is None or window.ended_by != self._shift_window.ended_by
        ):
            self._log(
                "shift",
                f"ended by {window.ended_by}" if window.ended_by else "window started",
                f"{window.adjust_c:+.1f} C until {window.end.isoformat()}",
            )
        self._shift_window = window
        self._save_state()

    def _open_fraction(self) -> float | None:
        """Share of the floor loops that are open now; None when no actuators are configured."""
        entities = [
            e for group in self._settings.room_actuators.values() for e in group
        ]
        if not entities:
            return None
        states = [self._ha.get_state(e)["state"] for e in entities]
        if any(state not in ("on", "off") for state in states):
            raise ValueError("an actuator is unavailable")
        return sum(state == "on" for state in states) / len(states)

    def _room_level(self, now: datetime, room_c: float) -> float:
        """The room temperature the plan starts from (see room_level)."""
        if self._actuals_loader is None:
            return room_c
        end = _floor_hour(now.astimezone(UTC))
        try:
            recent = self._actuals_loader(end - timedelta(hours=24), end).room_c
        except TheriaError:
            logger.warning("could not read the room history", exc_info=True)
            return room_c
        return room_level_c(room_c, recent)

    def tuning(self) -> list[Recommendation]:
        """Advice on the settings the plan never changes by itself (nothing is applied)."""
        now = self._now()
        if self._outcome_loader is None:
            days: list[DayOutcome] = []
        else:
            cached = self._tuning_days
            if cached is None or now - cached[0] > TUNING_CACHE:
                today = datetime.combine(now.date(), datetime.min.time(), UTC)
                loaded = self._outcome_loader(
                    today - timedelta(days=TUNING_DAYS), today
                )
                self._tuning_days = (now, loaded)
            days = self._tuning_days[1] if self._tuning_days else []
        latest = self.latest()
        return build_recommendations(
            self._settings,
            days,
            [] if latest is None else latest.plan.hours,
            self._night,
            authority=self._authority(),
        )

    @property
    def timezone(self) -> str:
        return self._settings.timezone

    def _periods_between(self, start: datetime, stop: datetime) -> list[PricePeriod]:
        """The feed's periods in [start, stop): today and tomorrow (a feed that keeps older days has its own history)."""
        return [p for p in self._price_periods() if start <= p.start < stop]

    def day_view(self, day: date) -> dict[str, Any]:
        """A whole local day for the charts: measured so far, planned for the rest."""
        now = self._now()
        latest = self.latest()
        measured = None
        zones = None
        first = day_hours(day, self._settings.timezone)[0]
        current = _floor_hour(now.astimezone(UTC))
        if self._actuals_loader is not None and first < current:
            try:
                # a day of lead-in: the setpoint is only logged when it changes
                measured = self._actuals_loader(first - timedelta(days=1), current)
                if self._zone_loader is not None:
                    zones = self._zone_loader(first, current)
            except TheriaError:
                logger.warning("could not read today's measurements", exc_info=True)
        salon = self._salon_reading()
        hours = day_hours(day, self._settings.timezone)
        periods = self._price_history(hours[0], hours[-1] + timedelta(hours=1))
        records = (
            {}
            if self._plan_record is None
            else self._plan_record.read(hours[0], hours[-1] + timedelta(hours=1))
        )
        return build_day_view(
            day,
            now,
            self._settings,
            periods,
            None if latest is None else latest.plan,
            measured,
            records,
            zones,
            None if salon is None else salon.mean_c,
        )

    def history(self, hours: int) -> list[PlanSnapshot]:
        cutoff = self._now() - timedelta(hours=hours)
        return [s for s in self._snapshots if s.computed_at >= cutoff]

    def model_status(self) -> dict[str, Any]:
        fitted = self._models
        return {
            "pump": {
                "source": "prior" if fitted is None else "fitted",
                "params": self._pump,
                "fit": None if fitted is None else fitted.pump,
            },
            "room": {
                "source": "fitted" if fitted is not None and fitted.room else "prior",
                "params": self._room,
                "fit": None if fitted is None else fitted.room,
                "fit_error": None if fitted is None else fitted.room_error,
            },
            "night": {
                "source": "none" if self._night is None else "fitted",
                "fit": self._night,
                "fit_error": self._night_error,
            },
            "warmup": {
                "source": "none" if self._warmup is None else "fitted",
                "fit": self._warmup,
                "fit_error": self._warmup_error,
            },
            "fitted_at": None if fitted is None else fitted.fitted_at,
            "window": (
                None
                if fitted is None
                else {"start": fitted.window_start, "end": fitted.window_end}
            ),
            "last_fit_error": self._last_fit_error,
        }

    def scores(self) -> list[LeadScore]:
        """Room temperature prediction error by lead time, from the stored snapshots."""
        return self._scoreboard.scores(list(self._snapshots))

    def run_once(self) -> PlanSnapshot:
        """Compute and store one plan."""
        now = self._now()
        s = self._settings

        periods = self._price_periods()
        hourly = hourly_prices(periods, _floor_hour(now))
        starts = [h for h, _ in hourly]
        prices = np.array([p for _, p in hourly])

        weather = {w.start: w for w in self._weather.fetch()}
        measured_outdoor = self._read_float(s.outdoor_entity, OUTDOOR_PLAUSIBLE_C)
        bias = measured_outdoor - self._forecast_at(weather, starts[0]).temperature_c
        t_out = np.array(
            [self._forecast_at(weather, h).temperature_c + bias for h in starts]
        )
        sun = np.array([self._forecast_at(weather, h).shortwave_wm2 for h in starts])
        warmup = [
            self._forecast_at(weather, starts[0] - timedelta(hours=k)).shortwave_wm2
            for k in range(SUN_WARMUP_HOURS, 0, -1)
        ]
        sun_filtered_now = float(
            sun_filtered_sequence(self._room, 0.0, np.array([*warmup, 0.0]))[-1]
        )

        room_c = (
            self._sensor_guard.room_temperature()
            if self._sensor_guard is not None
            else float(np.mean([self._read_float(e) for e in s.room_temp_entities]))
        )
        self._scoreboard.record(now, room_c)
        flow_now = self._read_float(s.flow_setpoint_entity, FLOW_PLAUSIBLE_C)
        previous_rts = float(
            self._ha.get_climate_state(s.room_setpoint_entity)["target_temperature"]
        )
        swing_now = self._room.sun_swing_c_per_wm2 * sun_filtered_now
        self._sample_disturbance(now, room_c - swing_now, weather)
        # with the sun's swing known the level is the measured room minus the swing; without it the
        # lower of the room and its day's median stands in for it
        level_c = (
            room_c - swing_now
            if self._room.sun_swing_c_per_wm2 > 0
            else self._room_level(now, room_c)
        )
        initial = RoomState(
            room_c=level_c,
            slab_c=level_c + SLAB_INIT_FRACTION * (flow_now - level_c),
            sun_filtered_wm2=sun_filtered_now,
        )

        inputs = PlanInputs(
            hour_starts=starts,
            price_sek_per_kwh=prices,
            t_out_c=t_out,
            sun_wm2=sun,
            initial=initial,
            previous_rts_c=previous_rts,
            extra_drift_c_per_h=self._extra_drift(starts),
            past_prices_sek_per_kwh=self._prices_today(periods, starts[0]),
            shift=self._shift_readings(room_c),
            **self._energy_table(starts, weather, bias),
        )
        room_params = self._warmed_room()
        effective = self._effective_settings()
        result = plan_shift(inputs, effective, self._pump, room_params)
        self._keep_shift_window(result.shift_window)
        self._keep_run(now, inputs, effective, room_params, result)
        snapshot = PlanSnapshot(
            computed_at=now,
            mode=self._mode,
            plan=result,
            outdoor_bias_c=bias,
            initial=initial,
            comfort_outlook=self._comfort_outlook(result, effective, room_params),
            model_status={
                "energy": (
                    "pump model" if inputs.energy_kwh_table is None else "forecast"
                ),
                "pump": "prior" if self._models is None else "fitted",
                "room": "fitted" if self._models and self._models.room else "prior",
            },
        )
        self._snapshots.append(snapshot)
        self._record_plan(snapshot)
        return snapshot

    def _keep_run(
        self,
        now: datetime,
        inputs: PlanInputs,
        settings: PlanSettings,
        room: RoomParams,
        result: Plan,
    ) -> None:
        """Keep what the plan was computed from, so that it can be replayed."""
        if self._plan_runs is None:
            return
        try:
            self._plan_runs.append_run(
                make_run(now, self._mode, inputs, settings, self._pump, room, result)
            )
        except OSError:
            logger.warning("could not write the plan run", exc_info=True)

    def plan_runs(self, since: datetime) -> list[dict[str, Any]]:
        return [] if self._plan_runs is None else self._plan_runs.read(since)

    def _record_plan(self, snapshot: PlanSnapshot) -> None:
        """Keep the first plan of each hour: the one that is applied for it."""
        if self._plan_record is None or not snapshot.plan.hours:
            return
        hour, flat = snapshot.plan.hours[0], snapshot.plan.baseline_hours[0]
        try:
            self._plan_record.append(
                PlanRecord(
                    hour=hour.start,
                    computed_at=snapshot.computed_at,
                    rts_c=hour.rts_c,
                    room_c=hour.room_c_end + hour.sun_swing_c,
                    energy_kwh=hour.energy_kwh,
                    baseline_rts_c=flat.rts_c,
                    baseline_room_c=flat.room_c_end + flat.sun_swing_c,
                    baseline_energy_kwh=flat.energy_kwh,
                    price_sek_per_kwh=hour.price_sek_per_kwh,
                    flow_c=hour.flow_target_c,
                    baseline_flow_c=flat.flow_target_c,
                    binding=hour.reason.get("binding"),
                    thermostat=hour.thermostat,
                )
            )
        except OSError:
            logger.warning("could not write the plan record", exc_info=True)

    @staticmethod
    def _forecast_at(
        weather: dict[datetime, WeatherHour], hour: datetime
    ) -> WeatherHour:
        try:
            return weather[hour]
        except KeyError as e:
            raise WeatherDataError(f"no weather forecast for {hour.isoformat()}") from e

    async def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._mode == "active" and self._executor is not None:
            # confirming the thermostats takes up to 20 s and the supervisor may not wait for it:
            # what is saved first already says that the base is what the executors wrote
            self._executor.expect_base()
            if self._lk is not None:
                self._lk.expect_base()
            self._save_state()
            failure = await asyncio.to_thread(self._restore_locked)
            if failure is not None:
                logger.error("base setpoints not restored at shutdown: %s", failure)
            # what the executors now remember is the base they just wrote: without this the next
            # start finds the base where it last wrote a plan value and takes it for a hand change
            self._save_state()

    def _restore_locked(self) -> str | None:
        with self._lock:
            return self._restore_all()

    async def cycle(self) -> None:
        """One supervisor pass: refit when due, then plan. Never raises."""
        if self.refit_due():
            try:
                await asyncio.to_thread(self.refit)
                logger.info("plan models refitted")
            except TheriaError:
                logger.exception("plan model refit rejected; keeping previous models")
            except Exception as e:  # supervisor boundary: the loop must survive
                self._schedule_refit_retry(e)
                logger.exception("plan model refit failed; keeping previous models")
        if self._energy_forecast is not None:
            try:
                await asyncio.to_thread(self._energy_forecast.run_due)
            except Exception:  # supervisor boundary: the loop must survive
                logger.exception("the energy forecast failed")
        try:
            await asyncio.to_thread(self.run_once)
            self._plan_error = None
        except Exception as e:
            self._plan_error = f"{type(e).__name__}: {e}"
            logger.exception("plan computation failed")
        if self._forecast_check is not None:
            try:
                await asyncio.to_thread(self._forecast_check.run_due)
            except Exception:  # supervisor boundary: the loop must survive
                logger.exception("checking the plans against what happened failed")
        if self._mode == "active":
            try:
                await asyncio.to_thread(self._control)
            except Exception:
                logger.exception("plan control failed")

    def _plan_hour_now(self, now: datetime) -> HourPlan | None:
        snapshot = self.latest()
        if snapshot is None:
            return None
        return next(
            (
                h
                for h in snapshot.plan.hours
                if h.start <= now < h.start + timedelta(hours=1)
            ),
            None,
        )

    def planned_setpoint(self, hour: datetime) -> float | None:
        """The setpoint the plan will write in `hour`; None when it writes nothing (shadow mode)."""
        if self._mode != "active":
            return None
        plan_hour = self._plan_hour_now(hour)
        return None if plan_hour is None else plan_hour.rts_c

    def _trace_cycle(
        self,
        pump: ExecutionResult,
        lk: LKResult | None,
        holds: dict[str, float] | None,
    ) -> None:
        """Keep what this cycle read, asked for and wrote (see exec_trace)."""
        if self._exec_trace is None or self._executor is None:
            return
        now = self._now()
        hour = self._plan_hour_now(now)
        effective = self._effective_settings()
        try:
            self._exec_trace.append(
                {
                    "at": now.isoformat(),
                    "mode": self._mode,
                    "pump": {
                        "read": self._executor.last_read,
                        "result": {
                            "action": pump.action,
                            "target_c": pump.target_c,
                            "written_c": pump.written_c,
                            "reason": pump.reason,
                        },
                    },
                    "lk": (
                        None
                        if lk is None or self._lk is None
                        else {
                            "read": self._lk.last_read,
                            "holds": holds,
                            "targets": self._lk.last_targets,
                            "result": {
                                "action": lk.action,
                                "target_c": lk.target_c,
                                "reason": lk.reason,
                            },
                        }
                    ),
                    "plan_hour": (
                        None
                        if hour is None
                        else {
                            "start": hour.start.isoformat(),
                            "rts_c": hour.rts_c,
                            "closed": hour.valves_closed,
                            "floor_c": hour.floor_c,
                            "price_sek_per_kwh": hour.price_sek_per_kwh,
                            "binding": hour.reason.get("binding"),
                        }
                    ),
                    "rts_range_c": [effective.rts_min_c, effective.rts_max_c],
                    "disturbance_c_per_h": self._disturbance.at(now),
                }
            )
        except OSError:
            logger.warning("could not write the execution trace", exc_info=True)

    def _entity_snapshot(self) -> dict[str, Any]:
        """What Home Assistant shows for every entity the settings name."""
        ids = sorted(
            {e for e in _strings(asdict(self._settings)) if ENTITY_ID.match(e)}
        )
        out: dict[str, Any] = {}
        for entity in ids:
            try:
                state = self._ha.get_state(entity)
                out[entity] = {
                    "state": state["state"],
                    "attributes": state["attributes"],
                }
            except (RuntimeError, ValueError, KeyError, TypeError) as e:
                out[entity] = {"error": f"{type(e).__name__}: {e}"}
        return out

    def debug_report(self, compact: bool = True) -> str:
        """Everything needed to understand what Theria did, as markdown (see debug_export)."""
        now = self._now()
        since = now - timedelta(hours=24 if compact else 168)
        runs = self.plan_runs(since)
        trace = self.exec_trace(since)
        events = self.action_log()
        preflight = [(c.name, c.status, c.detail) for c in self.preflight()]
        local = datetime.now()
        logs = read_log_lines(
            self._log_dir,
            [f"{local - timedelta(days=1):%Y-%m-%d}", f"{local:%Y-%m-%d}"],
        )
        return build_report(
            now=now,
            version=os.environ.get("BUILD_VERSION", "unknown"),
            mode=self._mode,
            findings=key_findings(
                runs, trace, events, logs if isinstance(logs, list) else [], preflight
            ),
            preflight=preflight,
            settings={
                "plan": asdict(self._settings),
                "effective_rts_range_c": [
                    self._effective_settings().rts_min_c,
                    self._effective_settings().rts_max_c,
                ],
                "catalog": self.settings_catalog(),
            },
            execution=self.execution_status(),
            entities=self._entity_snapshot(),
            runs=runs,
            trace=trace,
            events=events,
            logs=logs,
            compact=compact,
        )

    def exec_trace(self, since: datetime) -> list[dict[str, Any]]:
        return [] if self._exec_trace is None else self._exec_trace.read(since)

    def _room_c(self, thermostat: str) -> float | None:
        try:
            return float(self._ha.get_climate_state(thermostat)["current_temperature"])
        except (RuntimeError, ValueError, KeyError, TypeError):
            logger.warning("cannot read the room at %s", thermostat, exc_info=True)
            return None

    def _shift_holds(self) -> tuple[dict[str, float], str | None]:
        """Every thermostat's target in the heat shift, and the window it belongs to (None outside one)."""
        s = self._settings
        now = self._now()
        snapshot = self.latest()
        normal = dict.fromkeys(s.room_thermostat_entities, s.lk_base_c)
        if (
            self._restore_failed
            or snapshot is None
            or now - snapshot.computed_at > MAX_PLAN_AGE
        ):
            return normal, None
        hours = snapshot.plan.hours
        current = next(
            (
                i
                for i, h in enumerate(hours)
                if h.start <= now < h.start + timedelta(hours=1)
            ),
            None,
        )
        if current is None or hours[current].thermostat == NORMAL:
            return normal, None
        mode = hours[current].thermostat
        last = current
        while last + 1 < len(hours) and hours[last + 1].thermostat == mode:
            last += 1
        night = s.in_night(now)
        targets = {
            entity: thermostat_target(
                mode, night, self._room_c(entity) if mode == OPENING else None, s
            )
            for entity in s.room_thermostat_entities
        }
        return targets, f"{mode}:{hours[last].start.isoformat()}"

    def _control(self) -> None:
        """Active mode: review the last days once a day, watch for trouble, apply the plan."""
        if self._executor is None:
            raise ConfigurationError("active mode without an executor")
        with self._lock:
            if self._mode != "active":  # switched off while this cycle was waiting
                return
            today = self._now().date()
            try:
                self._review_days(today)
            except Exception:  # a failing history must not leave the pump uncontrolled
                logger.exception("daily review failed; control continues")
            if self._automation_conflict():
                return
            current = self._current_setpoint()
            base = self._settings.rts_base_c
            deviated = current is None or abs(current - base) > 1e-9
            raised = current is not None and current > base + 1e-9
            trouble = None if self._safety is None else self._safety.check(raised)
            if trouble is not None and deviated:
                self._last_trouble = trouble
                self._log("trouble", "reported", trouble)
                self._notify(TROUBLE_NOTIFICATION, "Theria: heater running", trouble)
            pump = self._executor.execute(self.latest())
            if pump.action != "kept":
                self._log("pump", pump.action, f"{pump.target_c}: {pump.reason}")
            holds = None
            lk = None
            if self._lk is not None:
                holds, window = self._shift_holds()
                lk = self._lk.execute(holds, window)
                if lk.action != "kept":
                    self._log("lk", lk.action, f"{lk.target_c}: {lk.reason}")
            self._trace_cycle(pump, lk, holds)
            self._save_state()

    def _automation_conflict(self) -> bool:
        entity = self._settings.conflicting_automation_entity
        try:
            conflict = self._automation_is_on()
        except (RuntimeError, ValueError, KeyError, TypeError):
            return (
                False  # HA unreadable: the executor's own read will fail the same way
            )
        if conflict:
            self._last_trouble = f"{entity} is on; the plan is not writing the setpoint"
        return conflict

    def _current_setpoint(self) -> float | None:
        try:
            return float(
                self._ha.get_climate_state(self._settings.room_setpoint_entity)[
                    "target_temperature"
                ]
            )
        except (RuntimeError, ValueError, KeyError, TypeError):
            return None

    def _review_days(self, today: date) -> None:
        if self._reviewed_on == today or self._outcome_loader is None:
            return
        start = datetime.combine(
            today - timedelta(days=OUTCOME_LOOKBACK_DAYS), datetime.min.time(), UTC
        )
        stop = datetime.combine(today, datetime.min.time(), UTC)
        outcomes = self._outcome_loader(start, stop)
        self._reviewed_on = today
        self._learn_morning_margin(outcomes, today)
        self._save_state()
        yesterday = next(
            (
                d
                for d in outcomes
                if d.day == today - timedelta(days=1) and d.coverage > 0
            ),
            None,
        )
        if yesterday is not None:
            self._notify(
                DAILY_NOTIFICATION,
                "Theria: yesterday",
                f"{yesterday.day}: saving {yesterday.saving_sek:.0f} SEK "
                f"(cost {yesterday.actual_cost_sek:.0f} SEK against "
                f"{yesterday.baseline_cost_sek:.0f} SEK at the base setpoint). "
                f"Room {yesterday.room_min_c:.1f}-{yesterday.room_mean_c:.1f} C (min, mean), "
                f"electric heater {yesterday.aux_kwh:.1f} kWh.",
            )

    def _notify(self, notification_id: str, title: str, message: str) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.notify(notification_id, title, message)
        except Exception:  # a failed notification must not stop the control loop
            logger.exception("could not send notification %s", notification_id)

    def wake(self) -> None:
        """Run the next cycle now (after a change of mode), not at the end of the interval."""
        self._wake.set()

    async def _loop(self) -> None:
        while True:
            self._wake.clear()
            await self.cycle()
            try:
                await asyncio.wait_for(self._wake.wait(), self._interval_seconds)
            except TimeoutError:
                pass
