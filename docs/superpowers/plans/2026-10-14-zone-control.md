# Zone control (v2) Implementation Plan

> For agentic workers: execute task by task, test first (watch it fail), commit per task.

**Goal:** Implement `docs/superpowers/specs/2026-10-14-zone-control-design.md`: the flow temperature is the
plan's lever, the LK thermostats are static per-room limits, and a 15-minute regulator measures and corrects
on the measured room temperatures.

**Architecture:** Build on the existing stack (`planner`, `room_model`, `calibration`, `night_drift`,
`sensor_guard`, `PlanService`). The planner stays one room-level state (the mean of the ground-floor rooms);
the per-room logic lives in the regulator, which is where the limits are.

**Spec:** `docs/superpowers/specs/2026-10-14-zone-control-design.md`

## Rulings (deviations from the spec, with reasons)

1. The planner keeps one room-level state, not one per room. Two rooms would make the DP four-dimensional;
   the per-room limits are enforced by the regulator and by the learned overshoot margin on the level.
2. `a_i`, `b_i` stay on the existing daily response + night drift until there are enough nights with the
   actuator history (it only exists since 2026-10-13); the hourly night fit is added with a minimum-data guard
   and is inactive until the guard passes.
3. Historical fits are not capped at `floor_mix_max_c`: it is not known when the shunts were set to 40 C.
   The cap is applied in forward planning only.
4. The radiator lower bound is the simple rule "no cuts below the base setpoint while a radiator calls for
   heat" first; learning a flow per outdoor band is left in TODO.
5. Authority by skill replaces the calendar stages: a factor in [0.5, 1] scales the allowed swing around the
   base setpoint; it starts at 0.5 and grows with measured no-sun prediction skill.
6. The new control logic is deployed in shadow mode and verified there before active mode is restored.

## Tasks

### Task 1: Settings - one LK limit, the shunt cap, ceiling 22
- Modify `core/theria/plan/settings.py`: remove `lk_setpoint_c`/`lk_base_c`; add `lk_limit_c: float = 22.0`
  (static LK setpoint = the room limit `U`), `floor_mix_max_c: float = 40.0`; `ceiling_c` default 22.0;
  validation `floor_c <= lk_limit_c <= ceiling_c` and `floor_mix_max_c > 0`.
- Update users: `planner.py`, `service.py`, `lk_executor.py`, `config.yaml` (options + schema), tests.
- Test: settings validation; defaults.

### Task 2: Model - the floor sees `min(flow, floor_mix_max_c)`
- `planner.py`: cap the flow fed to the room step in `transition`, `flat_plan`, `deficit_cost_sek_per_c`.
- Test: with a flow above the cap, raising the setpoint does not change the room's predicted rise but still
  costs electricity, so a flat-price plan lowers the setpoint (the lever is free to cut).

### Task 3: LKExecutor - static limit, no boost
- `lk_executor.py`: `execute(holds)` writes `lk_limit_c` to every thermostat; `holds` maps a thermostat to a
  setpoint that closes it (written by the regulator); hand change still pauses 1 h; `restore_base` writes the
  limit. Remove pre-heating logic.
- Tests: static write; hold overrides; hand change pause; failed write reported.

### Task 4: Disturbance estimator replaces the observer
- New `core/theria/plan/disturbance.py`: one-step innovation from measured inputs
  (`room_model.step` with dt = elapsed hours, measured flow capped at the shunt, measured open fraction, outdoor,
  filtered sun), smoothed with gain and fading like the observer; `correction(starts, now)` keeps the API.
- `service.py`: sample inputs each cycle, use it for `_extra_drift`; delete `observer.py` and its tests.
- Tests: a room that warms 0.1 C/h more than the model with the same inputs yields a positive estimate that
  fades; a plan's own effect (valves, flow) is not read as drift.

### Task 5: Per-room regulator
- New `core/theria/plan/regulator.py`: room temperature = mean of valid sensors per room; learned overshoot
  `o_i` from events (valves close at the limit -> peak - limit, slow average, bounded); decision per room
  (close / reopen with hysteresis; anticipate only when the projected peak passes `U + o_i`); shadow or active;
  every decision logged.
- `service.py`: run in each cycle, pass holds to the LK executor in active mode, expose status.
- `settings.py`: `room_controls` list of `{room, thermostat_entity, temperature_entities}`; `regulator_mode`.
- Tests: closes a room over its limit while the other stays open; reopens with hysteresis; learns an overshoot;
  shadow mode logs and writes nothing.

### Task 6: Sun term
- New `core/theria/plan/sun_fit.py`: `k_sun` from hourly daytime residuals (with a minimum number of sunny
  hours, bounds, bounded change per day); `calibration.py` stores it; `room_params_from_fit` carries it.
- Tests: recovers a known `k_sun` from synthetic data; refuses with too few sunny hours.

### Task 7: Authority from skill
- New `core/theria/plan/skill.py`: rolling mean absolute error of the hourly room-change prediction in no-sun
  hours; `authority` in [0.5, 1]; `_effective_settings` narrows `rts_min_c`/`rts_max_c` around the base.
- Tests: no history -> 0.5; good skill -> grows; bad skill -> back to 0.5.

### Task 8: Radiator calls
- `settings.py`: `radiator_entities`; `service.py` reads `hvac_action`/temperatures; while any radiator calls,
  `rts_min_c` is the base. Tests: calling -> no cuts; idle -> cuts allowed.

### Task 9: API and UI
- `execution_status()` gains `regulator`, `authority`, `radiators`; Control view panel per room (temperature,
  limit, valves, overshoot, disturbance), Models view skill. Tests: model tests for the new fields.

### Task 10: Docs and verification
- Update the knowledge file, TODO and the spec with what was built; deploy; verify in Chrome (shadow first),
  then restore active mode and watch the cycles.
