# Salon and guest house radiator control (shadow release) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute, log and show (never write) a price-driven setpoint plan for the salon (Netatmo) and the guest house (Tado) radiators, with a schedule and room settings editable in the Theria UI.

**Architecture:** A new package `core/theria/radiators/` holds pure planning functions (salon: small dynamic program on the learned thermal model; guest house: price rank within a band), a JSON store for the editable preferences, and a `RadiatorService` that runs every 15 minutes in shadow mode and logs what it would write. `backend/radiator_api.py` exposes it; `frontend/src/views/Radiators.tsx` shows and edits it. The pump plan is not changed in this release.

**Tech Stack:** Python 3.11+, FastAPI, numpy, pytest; React/Vite/vitest frontend; Home Assistant via `HAClient`.

**Spec:** `docs/superpowers/specs/2026-11-01-radiator-zones-design.md` (sections 1-13; section 13 holds the decisions made after the builder check).

## Global Constraints

- Shadow mode only: **nothing in this plan writes to Netatmo, Tado or the pump**. `mode` is `off | shadow`; `active` must raise `ConfigurationError`.
- Salon: weekdays 09:00-19:00, weekends 10:00-16:00 local time (Europe/Stockholm); mean of all valid points about 19.0 C while open (`salon_open_target_c` 19.0); closed floor 17.0 (`salon_closed_floor_c`); pre-heat 20.0 (`salon_preheat_c`); margin 0.3 (`salon_margin_c`); max setpoint 22.0 (`salon_max_setpoint_c`). Both thermostats get the same setpoint. Fewer than two valid points: zone "unknown", not cut.
- Guest house: per-room `T_room`, band `+/- 1.5 C` (global default, per-room override), setpoints in 0.5 C steps within Tado's 5-25 C; a room outside its band is never pushed further out.
- Netatmo range 7-30, Tado 5-25, both `target_temp_step` 0.5.
- Prices: 15-minute Nord Pool via `NordpoolFeed` (`price_periods`), hourly means via `hourly_prices`.
- Follow `CODING_STANDARDS.md`: no commit without Johan's approval, no AI attribution lines other than the harness's, no string matching on exceptions, no `hasattr`/fallbacks, deterministic, zero IDE errors. Run `ruff` and `mypy` as configured in `pyproject.toml`.
- Salon thermal model defaults (Dec 2025): C 2.33e6 J/K, U 51.7 W/K. Heating power and baseline are estimates and are settings.

## Review Focus

- Day-ahead prices not yet published (before about 13:00): the horizon is shorter than 36 h; the plan must still be valid and must not cut on a missing hour. Test in Task 4.
- DST change (2026-10-25, 25-hour day) and schedule edges (close at exactly 19:00): open flags must be per absolute hour. Test in Task 4.
- Salon sensors silent, `unavailable` or only one valid: zone unknown, setpoints unchanged (logged "unknown"), never a cut. Test in Task 3 and Task 7.
- Flat prices or a saving below the minimum: no movement, setpoints stay at today's values. Test in Task 4 and Task 5.
- A guest-house room already below its band when prices are dear: the lower bound wins. Test in Task 5.
- Thermostat entity `unavailable` or `climate` attributes missing: that thermostat is skipped and counted in the availability log, the cycle does not raise. Test in Task 7.

---

## File structure

| File | Responsibility |
|---|---|
| `core/theria/radiators/__init__.py` | package marker |
| `core/theria/radiators/settings.py` | `RadiatorSettings` (entities, physical constants, from `config.yaml`) and `RadiatorPrefs` (the editable values) |
| `core/theria/radiators/store.py` | `PrefsStore`: load/save/validate the editable preferences as JSON |
| `core/theria/radiators/salon_reading.py` | the salon's mean temperature from valid points |
| `core/theria/radiators/salon_plan.py` | hourly salon plan (dynamic program) and setpoint mapping |
| `core/theria/radiators/guest_plan.py` | hourly guest-house setpoint per room |
| `core/theria/radiators/savings.py` | indicative saving, would-lift-block flag |
| `core/theria/radiators/service.py` | `RadiatorService`: 15-minute shadow cycle, log, status |
| `backend/radiator_api.py` | REST endpoints |
| `backend/app.py`, `config.yaml` | wiring and options |
| `frontend/src/views/Radiators.tsx`, `frontend/src/radiators.ts`, `frontend/src/api.ts`, `frontend/src/App.tsx` | UI |
| `tests/core/theria/radiators/`, `tests/backend/test_radiator_api.py`, `frontend/src/radiators.test.ts` | tests |

---

### Task 1: Settings and preferences

**Files:**
- Create: `core/theria/radiators/__init__.py`, `core/theria/radiators/settings.py`
- Modify: `config.yaml` (add `radiators` options and schema)
- Test: `tests/core/theria/radiators/__init__.py`, `tests/core/theria/radiators/test_settings.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class DaySchedule: open_at: time; close_at: time` (`open_at < close_at`).
  - `@dataclass(frozen=True) class GuestRoom: entity: str; t_room_c: float; band_c: float | None = None` (`None` = use the global band).
  - `@dataclass(frozen=True) class RadiatorPrefs` with fields `salon_schedule: dict[str, DaySchedule | None]` (keys `mon`..`sun`, `None` = closed all day), `salon_open_target_c=19.0`, `salon_closed_floor_c=17.0`, `salon_preheat_c=20.0`, `salon_margin_c=0.3`, `salon_max_setpoint_c=22.0`, `guest_band_c=1.5`, `guest_rooms: tuple[GuestRoom, ...]`, `mode: str = "shadow"`. Methods: `is_open(self, moment: datetime, tz: str) -> bool`; `to_dict() -> dict`; `@classmethod from_dict(data: dict) -> RadiatorPrefs`. `__post_init__` raises `ConfigurationError` unless `closed_floor <= open_target` and `open_target + margin <= preheat <= max_setpoint`, band in (0, 5], `mode in ("off","shadow")`.
  - `@dataclass(frozen=True) class RadiatorSettings`: `salon_thermostats: tuple[str, ...]`, `salon_sensors: tuple[str, ...]` (floor and ceiling points), `salon_sensor_guard: GuardSettings`, `outdoor_entity: str`, `timezone="Europe/Stockholm"`, `salon_c_j_per_k=2.33e6`, `salon_u_w_per_k=51.7`, `salon_heat_power_w=1200.0`, `salon_baseline_c=20.0`, `guest_loss_w_per_k=100.0`, `defaults: RadiatorPrefs`. `@classmethod from_dict(data: dict) -> RadiatorSettings`, `@classmethod from_options(options: dict) -> RadiatorSettings | None` (section `radiators`), `load_radiator_settings(options_path: str, config_path: str) -> RadiatorSettings | None`.

- [ ] **Step 1: Write the failing tests** in `test_settings.py`:
  `test_defaults_match_the_spec` (weekday 09:00-19:00, Saturday 10:00-16:00, 19.0/17.0/20.0/0.3/22.0, band 1.5); `test_is_open_at_the_edges` (09:00 open, 18:59 open, 19:00 closed, a Sunday 15:59 open and 16:00 closed, local time from UTC input); `test_is_open_uses_local_time_across_dst` (2026-10-25 is 25 hours long: 09:00 local is open whichever offset applies); `test_a_day_set_to_none_is_closed`; `test_invalid_preferences_raise_configuration_error` (preheat below open_target+margin, max setpoint below preheat, band 0, mode `active`); `test_prefs_round_trip_through_dict`; `test_unknown_keys_raise`.
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_settings.py -v`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `settings.py` as specified. Reuse `GuardSettings.from_dict` for `salon_sensor_guard`; reuse `load_options` from `core/theria/options.py`. Guest room defaults come from the config block (Step 4), not from code.
- [ ] **Step 4: Add the `radiators` section to `config.yaml`** (options and `schema`, same style as `plan`): `salon_thermostats: [climate.vantsidan, climate.klippsidan]`, `salon_sensors: [sensor.schamponeringstolen_temperature, sensor.butik_temperature_2]` (the second is the likely smoke-detector sensor, comment that it is to be confirmed), `salon_sensor_guard` with four sensors (`sensor.vantsidan_temperature`, `sensor.klippsidan_temperature` and the two above; room `salon`; `reference: true` on `sensor.vantsidan_temperature`), `outdoor_entity: sensor.outdoor`, and `defaults` with `mode: shadow` and guest rooms `climate.bedroom 18.0`, `climate.hallway 19.5`, `climate.kitchen 20.0`, `climate.tv_room_gh 19.5`, `climate.storage_room 17.0`. Add a test `test_config_yaml_radiators_section_loads` that parses the repo's `config.yaml` through `load_radiator_settings`.
- [ ] **Step 5: Run** the tests. Expected: PASS.
- [ ] **Step 6: Commit** (only after Johan approves commits): `feat: radiator zone settings and preferences`.

---

### Task 2: Preferences store

**Files:**
- Create: `core/theria/radiators/store.py`
- Test: `tests/core/theria/radiators/test_store.py`

**Interfaces:**
- Consumes: `RadiatorPrefs` (Task 1).
- Produces: `class PrefsStore: __init__(self, defaults: RadiatorPrefs, path: str | None)`; `current -> RadiatorPrefs` (property); `update(self, values: dict) -> RadiatorPrefs` (merges the given keys over the current prefs, validates through `RadiatorPrefs.from_dict`, saves atomically, raises `ConfigurationError` and changes nothing on invalid input); `reset(self) -> RadiatorPrefs`; `is_default(self) -> bool`. Mirror `SettingsManager` (`core/theria/plan/settings_manager.py`): a damaged file is ignored with a warning, save through a temporary file and `os.replace`.

- [ ] **Step 1: Write the failing tests:** `test_current_is_the_defaults_without_a_file`; `test_update_persists_and_a_new_store_reads_it`; `test_invalid_update_changes_nothing_and_raises`; `test_reset_returns_defaults_and_removes_the_override`; `test_a_damaged_file_is_ignored`; `test_a_partial_update_keeps_the_other_fields` (change only `salon_open_target_c`).
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_store.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `store.py`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: radiator preferences store`.

---

### Task 3: Salon reading

**Files:**
- Create: `core/theria/radiators/salon_reading.py`
- Test: `tests/core/theria/radiators/test_salon_reading.py`

**Interfaces:**
- Consumes: `SensorGuard.reports() -> list[SensorReport]` (`core/theria/sensor_guard.py`; fields `temperature_entity`, `value_c`, `usable`); the HA client's `get_climate_state(entity) -> dict` (`current_temperature`).
- Produces: `@dataclass(frozen=True) class SalonReading: mean_c: float | None; used: tuple[str, ...]; skipped: dict[str, str]; thermostat_mean_c: float | None; unknown: bool`; `read_salon(reports: list[SensorReport], salon_entities: Collection[str]) -> SalonReading`. The mean is the plain mean of **raw** values of usable reports (no correction to a reference: floor and ceiling intentionally differ). `unknown` is true when fewer than two points are usable; `mean_c` is then `None`. `thermostat_mean_c` is the mean of the usable thermostat-sensor points only (used for the setpoint bias in Task 4).

- [ ] **Step 1: Write the failing tests:** `test_mean_is_the_plain_mean_of_usable_points` (20.5, 20.0, 19.7, 19.1 gives 19.825); `test_a_silent_point_is_left_out_and_named` (`skipped` has its entity and the reason); `test_one_usable_point_is_unknown` (`unknown` true, `mean_c` None); `test_no_points_is_unknown`; `test_thermostat_mean_uses_only_the_thermostat_sensors`.
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_salon_reading.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `read_salon`. Build `SensorReport` fixtures in the tests the way `tests/core/theria/test_sensor_guard.py` does.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: salon mean from valid points`.

---

### Task 4: Salon plan

**Files:**
- Create: `core/theria/radiators/salon_plan.py`
- Test: `tests/core/theria/radiators/test_salon_plan.py`

**Interfaces:**
- Consumes: `RadiatorPrefs`, `RadiatorSettings` (Task 1); `step_up` from `core/theria/plan/lk_executor.py` (0.5 C steps).
- Produces:
  - `@dataclass(frozen=True) class SalonModel: c_j_per_k: float; u_w_per_k: float; heat_power_w: float`.
  - `@dataclass(frozen=True) class SalonHour: start: datetime; open: bool; heat: bool; mean_end_c: float; setpoint_c: float; price_sek_per_kwh: float; reason: str`.
  - `plan_salon(hours: list[datetime], prices: Sequence[float], t_out_c: Sequence[float], open_flags: Sequence[bool], start_mean_c: float, prefs: RadiatorPrefs, model: SalonModel, thermostat_bias_c: float, min_saving_sek: float = 0.2) -> list[SalonHour]`.
- Behaviour pinned by the spec:
  - Each hour is `heat` (power `heat_power_w`) or off; the mean evolves by `dT = (P*heat - U*(T - T_out)) * 3600 / C`, capped at `salon_preheat_c`. A dynamic program on a 0.1 C grid minimises `sum(price * P * heat)` subject to: end-of-hour mean `>= open_target + margin` in open hours, `>= closed_floor` otherwise; the horizon end is priced in energy (the energy to bring the mean back to `open_target`, at the horizon's mean price, credit for heat above it capped at 0.5 C), as in `planner.py`.
  - Setpoint per hour: heat hour = `min(max_setpoint, step_up(mean_end_c + thermostat_bias_c))`; off hour = the thermostat equivalent of the hour's lower bound minus 0.5 C (`step_down`), not below `closed_floor`. So an off hour cannot let the room fall below the bound: the thermostat restarts the radiator itself.
  - If the plan beats "hold the flat baseline" by less than `min_saving_sek`, or if no feasible plan exists, every hour is a flat hold (`reason` says why) and nothing is cut.
  - Hours with a missing price (NaN) are held, never cut.
  - `thermostat_bias_c` is clamped to `[0, 2]`.

- [ ] **Step 1: Write the failing tests** (build a small `SalonModel` with the Dec 2025 values and `heat_power_w=1200`):
  `test_flat_prices_hold_the_target_without_movement` (all `reason` mentions flat, setpoints identical);
  `test_a_dear_block_inside_opening_hours_is_coasted_after_a_pre_heat` (cheap hours then a price spike while open: the hour before the spike is `heat` with mean up to 20.0, the spike hours are off, every `mean_end_c >= 19.3` in open hours);
  `test_the_mean_never_falls_below_target_plus_margin_while_open`;
  `test_closed_hours_may_cool_down_to_but_not_below_the_closed_floor`;
  `test_the_room_is_back_at_target_when_opening` (a night with a dear early morning still reaches 19.3 at 09:00);
  `test_a_missing_price_hour_is_held` (NaN in the middle);
  `test_a_short_horizon_without_tomorrows_prices_is_valid` (12 hours only);
  `test_dst_day_has_open_flags_per_absolute_hour` (hourly UTC starts across 2026-10-25, the 25-hour day, and a normal Monday: `prefs.is_open` is true for 6 hours on the Sunday 10:00-16:00 and for 10 hours on Monday 09:00-19:00, whatever the UTC offset);
  `test_setpoint_follows_bias_and_is_capped` (bias 2.0 with a heat hour wanting 20.0 gives 22.0; never above `salon_max_setpoint_c`; always a multiple of 0.5);
  `test_a_tiny_saving_is_not_taken` (prices differing by an öre keep every hour a flat hold);
  `test_infeasible_requirements_hold_and_say_so` (start mean 12 C, 1 hour to open).
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_salon_plan.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `plan_salon` as above; `step_down` goes in the same module (mirror of `step_up`).
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: salon price plan`.

---

### Task 5: Guest house plan

**Files:**
- Create: `core/theria/radiators/guest_plan.py`
- Test: `tests/core/theria/radiators/test_guest_plan.py`

**Interfaces:**
- Consumes: `RadiatorPrefs`, `GuestRoom` (Task 1).
- Produces: `@dataclass(frozen=True) class GuestHour: start: datetime; setpoint_c: float; reason: str`; `plan_guest_room(hours: list[datetime], prices: Sequence[float], room: GuestRoom, band_c: float, room_temp_c: float | None, min_c: float = 5.0, max_c: float = 25.0) -> list[GuestHour]`.
- Behaviour: for each hour take the price's percentile rank `r` within the given prices (NaN hours excluded; ties share the mean rank); `f = 1 - 2r`; setpoint = `round_to_half(T_room + band * f)`, clamped to `[min_c, max_c]`. If `room_temp_c` is below `T_room - band` the setpoint is at least `T_room` (a room outside the band is never pushed further out); symmetrically above `T_room + band` it is at most `T_room`. If all prices are equal the setpoint is `T_room`. A missing price (NaN) gives `T_room`. `room_temp_c is None` skips the bound.

- [ ] **Step 1: Write the failing tests:** `test_cheapest_hour_is_the_top_of_the_band_and_dearest_the_bottom` (T_room 20, band 1.5: 21.5 and 18.5); `test_setpoints_are_half_degree_steps`; `test_equal_prices_give_t_room`; `test_a_room_below_its_band_is_not_cut` (room_temp 18.0, dear hour gives at least 20.0); `test_a_room_above_its_band_is_not_heated_further`; `test_per_room_band_overrides_the_global_one` (through the caller's `band_c`); `test_nan_price_gives_t_room`; `test_limits_clamp` (T_room 24.5 with band 1.5 stays <= 25.0).
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_guest_plan.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `plan_guest_room`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: guest house price plan`.

---

### Task 6: Savings estimate and the pump-block flag

**Files:**
- Create: `core/theria/radiators/savings.py`
- Test: `tests/core/theria/radiators/test_savings.py`

**Interfaces:**
- Consumes: `SalonHour`, `SalonModel` (Task 4); `GuestHour` (Task 5).
- Produces:
  - `salon_saving_sek(plan: list[SalonHour], baseline_c: float, t_out_c: Sequence[float], model: SalonModel) -> float`: heat energy of the baseline (hold the mean at `baseline_c`: `U*(baseline - T_out)` per hour, floored at 0) minus the plan's (`heat_power_w` for heat hours, counted with the model's partial-hour need), each times the hour's price, summed. Positive = saving.
  - `guest_saving_sek(plans: dict[str, list[GuestHour]], baseline_c: dict[str, float], prices: Sequence[float], loss_w_per_k: float) -> tuple[float, str]`: indicative, `loss_w_per_k/ n_rooms * (baseline - setpoint)` per room-hour times price, never negative per hour; returns the figure and the label `"indicative"`.
  - `would_lift_block(calling: Mapping[str, bool], controlled: Collection[str], readings_ok: bool) -> bool`: true only when every calling radiator is in `controlled` and `readings_ok` (all rooms at or above their lower bounds).

- [ ] **Step 1: Write the failing tests:** `test_a_plan_equal_to_the_baseline_saves_nothing`; `test_moving_heat_from_dear_to_cheap_hours_saves_money`; `test_guest_saving_is_labelled_indicative_and_zero_for_equal_setpoints`; `test_the_block_is_lifted_only_when_all_callers_are_controlled`; `test_the_block_is_not_lifted_when_a_room_is_below_its_bound`; `test_an_uncontrolled_caller_keeps_the_block`.
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_savings.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: radiator saving estimate`.

---

### Task 7: RadiatorService (shadow cycle)

**Files:**
- Create: `core/theria/radiators/service.py`
- Test: `tests/core/theria/radiators/test_service.py`

**Interfaces:**
- Consumes: Tasks 1-6; `HAClient.get_state`, `get_climate_state` (`core/theria/ha_client.py`); `PricePeriod`, `hourly_prices` (`core/theria/plan/prices.py`); `SensorGuard` (a second instance, built from `RadiatorSettings.salon_sensor_guard`); `EventLog` (`core/theria/plan/event_log.py`) for the persistent log.
- Produces: `class RadiatorService: __init__(self, ha, settings: RadiatorSettings, store: PrefsStore, guard: SensorGuard, price_periods: Callable[[], list[PricePeriod]], event_log: EventLog | None = None, now: Callable[[], datetime] = lambda: datetime.now(UTC), interval_seconds: int = 900)`; `run_once(self) -> RadiatorSnapshot | None` (None when `mode == "off"`); `latest(self) -> RadiatorSnapshot | None`; `async start()`, `async stop()`, `async cycle()` (never raises; mirror `PlanService.start/_loop/cycle`).
  - `@dataclass(frozen=True) class RadiatorSnapshot`: `computed_at`, `salon: SalonPlanView` (reading, hours, saving_sek), `guest: dict[str, list[GuestHour]]`, `would_write: dict[str, float]` (entity to this hour's setpoint), `read: dict[str, float | None]` (each thermostat's current setpoint), `would_lift_block: bool`, `availability: dict[str, bool]`, `guest_saving: tuple[float, str]`.
- Behaviour: reads prices, outdoor (`settings.outdoor_entity`, held flat over the horizon), the salon reading (guard `check()` then `read_salon`), each thermostat's `current_temperature`/`temperature`/`hvac_action`; plans the salon and each guest room; **never calls `set_temperature`** (a test asserts the fake HA recorded no service call). Unknown salon: `would_write` for the salon holds the current setpoints and the log line says "unknown". Unreadable thermostat: `availability[entity] = False`, skipped, no exception. One log line per cycle through `EventLog` with kind `radiators` and the would-be setpoints, the difference from the current setpoint, the salon mean and points used. `mode == "active"` cannot be constructed (settings already raise).

- [ ] **Step 1: Write the failing tests** with a fake HA (same style as `tests/core/theria/plan/test_service_radiators.py`) and fixed `now`:
  `test_a_cycle_makes_a_snapshot_with_a_would_be_setpoint_per_thermostat`;
  `test_a_cycle_never_writes_to_home_assistant` (fake records any `set_temperature`/`call_service`);
  `test_off_mode_returns_none_and_reads_nothing`;
  `test_unknown_salon_holds_the_current_setpoints` (only one usable point);
  `test_an_unavailable_thermostat_is_skipped_and_recorded` (availability false, other rooms planned, no exception);
  `test_the_log_records_the_difference_between_current_and_would_be_setpoint`;
  `test_missing_tomorrow_prices_still_give_a_plan`;
  `test_cycle_survives_a_failing_price_feed` (the error is stored, `cycle()` does not raise);
  `test_would_lift_block_is_reported_but_the_block_is_not_applied` (snapshot flag true while nothing else changes).
- [ ] **Step 2: Run** `pytest tests/core/theria/radiators/test_service.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** `service.py`; run blocking HA calls in `asyncio.to_thread` as `PlanService.cycle` does.
- [ ] **Step 4: Run** `pytest tests/core/theria/radiators -v`. Expected: all PASS.
- [ ] **Step 5: Commit** `feat: radiator shadow service`.

---

### Task 8: API and wiring

**Files:**
- Create: `backend/radiator_api.py`
- Modify: `backend/app.py` (start/stop next to the plan service, using the plan's `NordpoolFeed` as `price_periods`; second `SensorGuard` for the salon; overrides in `/data/radiator_prefs.json` with the same dev fallback as the other `/data` files; log `/data/radiator_events.jsonl`), include the router
- Test: `tests/backend/test_radiator_api.py`

**Interfaces:**
- Consumes: `RadiatorService`, `PrefsStore` (Tasks 2, 7).
- Produces (router prefix `/api/radiators`, dependency `get_service` like `plan_api.get_service`, 503 when not configured):
  - `GET /status`: the latest snapshot (404 until one exists) plus `age_seconds`.
  - `GET /prefs`: `{"prefs": ..., "defaults": ..., "is_default": bool}`.
  - `PUT /prefs` body `{"values": {...}}`: `store.update`; `ConfigurationError` becomes 422 with its message.
  - `DELETE /prefs`: reset.
  - `GET /log?limit=`: the persistent log.

- [ ] **Step 1: Write the failing tests** with `TestClient` and dependency overrides (as `tests/core/theria/plan/test_plan_api.py`): `test_status_is_404_until_a_cycle_has_run`; `test_status_returns_would_write_and_availability`; `test_prefs_roundtrip_through_put_and_get`; `test_invalid_prefs_are_422_with_a_message`; `test_delete_resets_to_defaults`; `test_503_when_not_configured`.
- [ ] **Step 2: Run** `pytest tests/backend/test_radiator_api.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** the router and wire it in `app.py`; start the service only when settings exist and `mode != "off"`; on shutdown `await service.stop()` (nothing to restore, the service never writes).
- [ ] **Step 4: Run** `pytest tests/backend tests/core -q`. Expected: PASS. Then `./test-server.sh` import check, then `./dev-start.sh` and `curl localhost:8081/api/radiators/status` (with `.env` set) returns a snapshot or 404 during the first cycle; confirm in the Home Assistant UI that the Netatmo/Tado setpoints are unchanged.
- [ ] **Step 5: Commit** `feat: radiator API and wiring`.

---

### Task 9: UI

**Files:**
- Create: `frontend/src/views/Radiators.tsx`, `frontend/src/radiators.ts`, `frontend/src/radiators.test.ts`
- Modify: `frontend/src/api.ts` (fetch/put/reset functions), `frontend/src/model.ts` (types), `frontend/src/App.tsx` (new tab `radiators`, label "Radiators")

**Interfaces:**
- Consumes: the API of Task 8.
- Produces (`radiators.ts`, pure): `scheduleToRows(prefs): ScheduleRow[]` (one row per weekday with `open`/`close` as `HH:MM` or `null`); `rowsToSchedule(rows): ScheduleValues` (validates `open < close`, returns an error string otherwise); `availabilityShare(log): number`; `formatDifference(current: number, wouldBe: number): string` (for example `+1.5` or `=`).

- [ ] **Step 1: Write the failing tests** in `radiators.test.ts`: `schedule round-trips weekdays and weekend`, `a closed day is null both ways`, `open must be before close`, `formatDifference shows sign and equality`, `availabilityShare counts readable cycles`.
- [ ] **Step 2: Run** `cd frontend && npm test`. Expected: FAIL.
- [ ] **Step 3: Implement** `radiators.ts`, then `Radiators.tsx` with, in the existing visual style (`Card`, `PageIntro`, `useLoaded` from `ui.tsx`): a status card ("Shadow: nothing is written"), the salon day (hourly price, planned mean, would-be setpoint, current setpoint), the guest rooms (T_room, band, would-be setpoint now, difference), the opening-hours editor, the numbers (target, closed floor, pre-heat, margin, max setpoint, band, per-room T_room and band), the saving with the "indicative" label for the guest house, availability, and the mode switch (off or shadow).
- [ ] **Step 4: Run** `cd frontend && npm test && npm run type-check && npm run build`. Expected: PASS and a built bundle. Open the page (run skill) and check the tab renders with live data and that saving a changed schedule persists after reload.
- [ ] **Step 5: Commit** `feat: radiators tab`.

---

### Task 10: Documentation

**Files:**
- Modify: `.claude/CLAUDE.md` (the "No frontend" line is out of date; add the radiator package to Important Files), `SYSTEM_SETUP.md` (salon sensors incl. Verisure points, Netatmo in `auto` with schedule `select.salong`, the new Tado/Netatmo facts), `documentation/TODO.md` (radiator control: shadow release done; write test; active mode; learn the flow per outdoor band; confirm the smoke-detector sensor), `documentation/ivt-heat-pump.md` (the salon's thermal numbers and that the radiators block pump cuts)

- [ ] **Step 1:** Make the edits, in the same style as the surrounding text. State open points as open: the Netatmo hold behaviour, Tado call limit, which sensor the ceiling point is.
- [ ] **Step 2:** `git diff --stat` shows only those four files. Run the full suite once more: `pytest -q` and `cd frontend && npm test`. Expected: PASS.
- [ ] **Step 3: Commit** `docs: radiator zone control (shadow)`.

---

## Self-review

- **Spec coverage:** S1 (Task 3), S2-S6 (Tasks 1, 4), G1-G4 (Tasks 1, 5), section 6 link to the pump plan in shadow (Tasks 6, 7: flag only), section 7 safety (shadow writes nothing; manual-change skipped in shadow per 13.7: Task 7 logs the difference), section 8 (Tasks 7-8), section 9 UI and settings (Tasks 1, 2, 8, 9), section 10 constraints (Task 1 settings, Task 8 no direct cloud access), section 13 (Tasks 4, 6, 7, 10). Not in this plan, by the spec: the manual write test, `active` mode, the pump-plan change that lifts the block.
- **Type consistency:** `RadiatorPrefs`, `GuestRoom`, `SalonModel`, `SalonHour`, `GuestHour`, `SalonReading`, `RadiatorSnapshot` are defined once and used with the same names later.
- **Known gap to watch at execution:** the salon heating power (`salon_heat_power_w` 1200) and baseline (20.0) are estimates; the first days of shadow data are for checking them, not for trusting the SEK figures.
