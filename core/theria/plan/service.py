"""Plan service: reads live data, computes the plan and keeps the latest snapshots.

Shadow mode never writes to Home Assistant. Active mode (explicitly confirmed) lets the executor
write the room setpoint within the current stage and steps back at the first sign of trouble.
"""

import asyncio
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol

import numpy as np

from core.theria.exceptions import (
    ConfigurationError,
    SensorError,
    SetpointRestoreError,
    TheriaError,
    WeatherDataError,
)
from core.theria.ha_client import HAClient
from core.theria.plan.calibration import Calibrator, FittedModels
from core.theria.plan.event_log import EventLog
from core.theria.plan.executor import ExecutionResult, PlanExecutor
from core.theria.plan.lk_executor import LKExecutor, LKResult
from core.theria.plan.morning import MorningRules, update_margin
from core.theria.plan.night_drift import NightDrift, drift_pattern
from core.theria.plan.outcomes import DayOutcome
from core.theria.plan.parameter_log import ParameterLog
from core.theria.plan.planner import PlanInputs, plan, sun_filtered_sequence
from core.theria.plan.preflight import Check, preflight_checks
from core.theria.plan.prices import (
    PriceHistory,
    extreme_threshold,
    hourly_prices,
    parse_nordpool_attributes,
)
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, PumpParams, flow_target
from core.theria.plan.room_model import (
    PRIOR_ROOM_PARAMS,
    RoomParams,
    RoomState,
    anchored_room_params,
)
from core.theria.plan.scoring import LeadScore, ScoreBoard
from core.theria.plan.settings import SUPPORTED_MODES, PlanSettings
from core.theria.plan.snapshot import ComfortOutlook, PlanSnapshot
from core.theria.plan.stages import (
    STAGES,
    StageRules,
    StageState,
    evaluate_stage,
    shrink_for_trouble,
)
from core.theria.plan.state_store import PlanState, PlanStateStore
from core.theria.plan.warmup import WarmupFit
from core.theria.plan.weather import OpenMeteoClient, WeatherHour
from core.theria.sensor_guard import SensorGuard

logger = logging.getLogger(__name__)

SLAB_INIT_FRACTION = 0.5  # rough prior: slab sits half way between room and flow target
SUN_WARMUP_HOURS = 24
LOG_KEPT = 200
SNAPSHOTS_KEPT = 96 * 2  # two days at one snapshot per 15 minutes


class Safety(Protocol):
    def check(self, raised: bool = False, deviated: bool = True) -> str | None: ...


class Notifier(Protocol):
    def notify(self, notification_id: str, title: str, message: str) -> None: ...

    def dismiss(self, notification_id: str) -> None: ...


OutcomeLoader = Callable[[datetime, datetime], list[DayOutcome]]
TROUBLE_NOTIFICATION = "theria_plan_trouble"
STAGE_NOTIFICATION = "theria_plan_stage"
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


class PlanService:
    """Computes the price-driven plan. Shadow mode: never writes to Home Assistant."""

    def __init__(
        self,
        ha_client: HAClient,
        settings: PlanSettings,
        weather: OpenMeteoClient,
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
        safety: Safety | None = None,
        notifier: Notifier | None = None,
        outcome_loader: OutcomeLoader | None = None,
        stage_rules: StageRules | None = None,
        parameter_log: ParameterLog | None = None,
        event_log: EventLog | None = None,
    ):
        self._ha = ha_client
        self._settings = settings
        self._weather = weather
        self._pump = pump_params
        self._room = room_params
        self._now = now
        self._interval_seconds = interval_seconds
        self._mode = settings.mode
        self._history = PriceHistory(settings.extreme_history_days)
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
        self._safety = safety
        self._notifier = notifier
        self._outcome_loader = outcome_loader
        self._rules = stage_rules or StageRules()
        self._stage = StageState(0, now().date(), None)
        self._stage_checked_on: date | None = None
        self._last_trouble: str | None = None
        self._last_stage_note: str | None = None
        # set_mode and the control cycle never interleave
        self._lock = threading.RLock()
        self._event_log = event_log
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
            if state.stage is not None:
                self._stage = state.stage
            self._morning_margin_c = state.morning_margin_c
            self._margin_changed_on = state.margin_changed_on
            self._stage_checked_on = state.stage_checked_on
            if state.executor is not None and executor is not None:
                executor.load_state(state.executor)
            if state.lk is not None and lk_executor is not None:
                lk_executor.load_state(state.lk)

    def now(self) -> datetime:
        return self._now()

    def _read_float(self, entity_id: str) -> float:
        """Numeric state of an entity; raises SensorError for unavailable/unknown."""
        raw = self._ha.get_state(entity_id)["state"]
        try:
            return float(raw)
        except (TypeError, ValueError) as e:
            raise SensorError(f"{entity_id} has non-numeric state {raw!r}") from e

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
                if failure is not None:
                    raise SetpointRestoreError(
                        f"the base setpoints could not be restored ({failure}); "
                        "the mode stays active"
                    )
            if mode == "active" and not was_active:
                today = self._now().date()
                revert = self._stage.revert_until
                self._stage = replace(
                    self._stage,
                    stage_started_on=today if revert is None else max(today, revert),
                )
                self._stage_checked_on = None
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
            "stage": self._stage.index + 1,
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
        baseline_flow: Any,
    ) -> ComfortOutlook | None:
        hours = result.hours
        for k in range(len(hours) - 1):
            if settings.in_night(hours[k].start) and not settings.in_night(
                hours[k + 1].start
            ):
                hour = hours[k]
                deficit = max(0.0, settings.target_c - hour.room_c_end)
                cap = min(
                    settings.max_flow_c,
                    max(settings.boost_flow_limit_c, float(baseline_flow[k])),
                )
                extra = room.k_slab * max(0.0, cap - hour.flow_target_c)
                lead = (
                    0.0 if deficit == 0.0 else (deficit / extra if extra > 0 else None)
                )
                return ComfortOutlook(
                    hour.start + timedelta(hours=1),
                    hour.room_c_end,
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

    def _effective_settings(self) -> PlanSettings:
        """The settings the plan runs on: the night kept warmer by the learned margin."""
        s = self._settings
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
        """What the plan did (oldest first): setpoint writes, trouble, stage and mode changes."""
        return list(self._events)

    def preflight(self) -> list[Check]:
        """What to read before activating: the state of everything active mode depends on."""
        fitted = self._models
        return preflight_checks(
            self._ha,
            self._settings,
            self.latest(),
            pump_fitted=fitted is not None,
            room_fitted=fitted is not None and fitted.room is not None,
            stage=self._stage,
            now=self._now(),
        )

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
                    stage=self._stage,
                    stage_checked_on=self._stage_checked_on,
                    morning_margin_c=self._morning_margin_c,
                    margin_changed_on=self._margin_changed_on,
                    executor=(
                        None
                        if self._executor is None
                        else self._executor.export_state()
                    ),
                    lk=None if self._lk is None else self._lk.export_state(),
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

    def execution_status(self) -> dict[str, Any]:
        stage = STAGES[self._stage.index]
        result: ExecutionResult | None = (
            None if self._executor is None else self._executor.last_result
        )
        lk_result: LKResult | None = None if self._lk is None else self._lk.last_result
        return {
            "mode": self._mode,
            "stage": {
                "index": self._stage.index,
                "started_on": self._stage.stage_started_on,
                "revert_until": self._stage.revert_until,
                "max_deviation_c": stage.max_deviation_c,
                "max_rise_per_hour_c": stage.max_rise_per_hour_c,
            },
            "stage_checked_on": self._stage_checked_on,
            "last_execution": result,
            "last_lk": lk_result,
            "morning_margin_c": self._morning_margin_c,
            "last_trouble": self._last_trouble,
            "last_stage_note": self._last_stage_note,
        }

    def _apply_models(self, models: FittedModels) -> None:
        self._models = models
        self._night = models.night_drift
        self._night_error = models.night_drift_error
        self._warmup = models.warmup
        self._warmup_error = models.warmup_error
        self._pump = models.pump.params
        if models.room is not None:
            self._room = models.room.params

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
        self._apply_models(models)
        self._record_parameters("refit", "nightly refit of the models")
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

    def run_once(self) -> PlanSnapshot | None:
        """Compute and store one plan; returns None when mode is off."""
        if self._mode == "off":
            return None
        now = self._now()
        s = self._settings

        attrs = self._ha.get_state(s.price_entity)["attributes"]
        periods = parse_nordpool_attributes(attrs)
        self._history.add(periods)
        self._history.prune(now)
        threshold = extreme_threshold(
            self._history.values(),
            s.extreme_price_percentile,
            s.extreme_price_min_sek_per_kwh,
        )
        hourly = hourly_prices(periods, _floor_hour(now))
        starts = [h for h, _ in hourly]
        prices = np.array([p for _, p in hourly])

        weather = {w.start: w for w in self._weather.fetch()}
        measured_outdoor = self._read_float(s.outdoor_entity)
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
        flow_now = self._read_float(s.flow_setpoint_entity)
        previous_rts = float(
            self._ha.get_climate_state(s.room_setpoint_entity)["target_temperature"]
        )
        initial = RoomState(
            room_c=room_c,
            slab_c=room_c + SLAB_INIT_FRACTION * (flow_now - room_c),
            sun_filtered_wm2=sun_filtered_now,
        )

        inputs = PlanInputs(
            hour_starts=starts,
            price_sek_per_kwh=prices,
            extreme=prices >= threshold,
            t_out_c=t_out,
            sun_wm2=sun,
            initial=initial,
            previous_rts_c=previous_rts,
            extra_drift_c_per_h=(
                None
                if self._night is None
                else drift_pattern(self._night, s.in_night, starts, t_out, room_c)
            ),
        )
        baseline_flow = np.asarray(
            flow_target(self._pump, t_out, s.rts_base_c), dtype=float
        )
        room_params = anchored_room_params(
            self._warmed_room(), room_c, t_out, baseline_flow, s.lk_setpoint_c
        )
        result = plan(inputs, self._effective_settings(), self._pump, room_params)
        snapshot = PlanSnapshot(
            computed_at=now,
            mode=self._mode,
            plan=result,
            extreme_threshold_sek_per_kwh=threshold,
            outdoor_bias_c=bias,
            initial=initial,
            comfort_outlook=self._comfort_outlook(
                result, self._effective_settings(), room_params, baseline_flow
            ),
            model_status={
                "pump": "prior" if self._models is None else "fitted",
                "room": "fitted" if self._models and self._models.room else "prior",
            },
        )
        self._snapshots.append(snapshot)
        return snapshot

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
            failure = await asyncio.to_thread(self._restore_locked)
            if failure is not None:
                logger.error("base setpoints not restored at shutdown: %s", failure)

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
        try:
            await asyncio.to_thread(self.run_once)
        except Exception:
            logger.exception("plan computation failed")
        if self._mode == "active":
            try:
                await asyncio.to_thread(self._control)
            except Exception:
                logger.exception("plan control failed")

    def _control(self) -> None:
        """Active mode: judge the stage once a day, watch for trouble, apply the plan."""
        if self._executor is None:
            raise ConfigurationError("active mode without an executor")
        with self._lock:
            if self._mode != "active":  # switched off while this cycle was waiting
                return
            today = self._now().date()
            try:
                self._judge_stage(today)
            except Exception:  # a failing history must not leave the pump uncontrolled
                logger.exception("stage judgement failed; control continues")
            if self._automation_conflict():
                return
            current = self._current_setpoint()
            base = self._settings.rts_base_c
            deviated = current is None or abs(current - base) > 1e-9
            raised = current is not None and current > base + 1e-9
            trouble = (
                None if self._safety is None else self._safety.check(raised, deviated)
            )
            if trouble is not None and deviated:
                self._restore_all()
                decision = shrink_for_trouble(
                    self._stage, today, self._rules, f"during the day: {trouble}"
                )
                self._stage = decision.state
                self._last_trouble = decision.reason
                self._log("trouble", "stepped back", decision.reason)
                self._record_parameters("stage", decision.reason)
                self._save_state()
                self._notify(
                    TROUBLE_NOTIFICATION, "Theria: plan stopped", decision.reason
                )
                return
            pump = self._executor.execute(self.latest(), self._stage)
            if pump.action != "kept":
                self._log("pump", pump.action, f"{pump.target_c}: {pump.reason}")
            if self._lk is not None:
                applied = self._current_setpoint()
                if applied is not None:
                    lk = self._lk.execute(applied)
                    if lk.action != "kept":
                        self._log("lk", lk.action, f"{lk.target_c}: {lk.reason}")
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

    def _judge_stage(self, today: date) -> None:
        if self._stage_checked_on == today or self._outcome_loader is None:
            return
        start = datetime.combine(
            today - timedelta(days=OUTCOME_LOOKBACK_DAYS), datetime.min.time(), UTC
        )
        stop = datetime.combine(today, datetime.min.time(), UTC)
        outcomes = self._outcome_loader(start, stop)
        decision = evaluate_stage(
            self._stage, outcomes, today, self._settings.target_c, self._rules
        )
        self._stage = decision.state
        self._stage_checked_on = today
        self._learn_morning_margin(outcomes, today)
        self._last_stage_note = f"{decision.action}: {decision.reason}"
        if decision.action != "hold":
            self._log("stage", decision.action, decision.reason)
            self._record_parameters("stage", decision.reason)
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
                f"electric heater {yesterday.aux_kwh:.1f} kWh, "
                f"compressor stopped {yesterday.compressor_stopped_hours} h.",
            )
        if decision.action == "widen":
            self._notify(
                STAGE_NOTIFICATION,
                "Theria: stage widened",
                f"Stage widened to +/-{STAGES[self._stage.index].max_deviation_c} C. "
                f"{decision.reason}",
            )
        elif decision.action == "shrink":
            self._notify(
                TROUBLE_NOTIFICATION, "Theria: plan stepped back", decision.reason
            )

    def _notify(self, notification_id: str, title: str, message: str) -> None:
        if self._notifier is None:
            return
        try:
            self._notifier.notify(notification_id, title, message)
        except Exception:  # a failed notification must not stop the control loop
            logger.exception("could not send notification %s", notification_id)

    async def _loop(self) -> None:
        while True:
            await self.cycle()
            await asyncio.sleep(self._interval_seconds)
