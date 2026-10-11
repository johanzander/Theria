# TODO - things to look at later

## First floor heating (current focus)

- [x] Quantify `room_temp_setpoint` effect on `heating_setpoint`: ~1.9 C per C, see knowledge base.
- [ ] Measure how fast first-floor temperatures respond to a flow-temperature change (concrete lag).
- [ ] Compare observed first-floor temperatures with the setpoint/outdoor history across seasons to size the
      lever Theria should use (curve shift vs LK setpoint).
- [ ] Expose LK actuator state (`actuators: [{n, s}]` per thermostat) as sensors in the `ha-lksystems` fork,
      so "2 of 5 loops open" can be seen in history. Verify what `s` values mean (0 assumed closed).
- [ ] Add a weather service (Open-Meteo, lat 55.578 / lon 12.927 from HA): hourly `shortwave_radiation`,
      `direct_radiation`, `cloud_cover`. Use as solar-gain term (Solcast PV forecast is a possible fallback).
- [x] Quantify solar gain per room from history (done, see knowledge base: ~2-3.5 C rise above ~2 kWh/m2/day, 4 h lag).
- [ ] Decide how to handle the winter undershoot (rooms 1-2.7 C below 19.5 C in Jan-Feb 2026): raise curve, raise room
      setpoint, or raise LK setpoint. Check which rooms/actuators limit it (Vardagsrum has 3 actuators, Bibliotek 5).
- [ ] Re-run the sun/lag analysis excluding evenings with stove use (infer: evening rise not explained by sun).
- [ ] Measure the concrete's response to a flow-temperature step with a controlled test (needs Johan's OK).
- [ ] Replace the 4 h guess with a thermal model per room (sun, outdoor, flow, LK valve state) once actuator states are logged.
- [ ] Find out what drives the hourly sawtooth in `radiator_forward` (DHW cycles? circulation pump?).
- [ ] Identify `sensor.heating_setpoint_2` (39 C) and `climate.external_control` / `external_control_2`.
- [x] CurveL/CurveR (H60 registers 0205/0206) are not used; the curve is set manually on the pump.
- [ ] Decide whether to feed a real indoor temperature through room-controller emulation (reg 02F1)
      instead of writing the room setpoint (H60 support unconfirmed).
- [ ] Check whether the system has a buffer tank / enough water volume for defrost when LK closes loops.
- [ ] Re-enable or replace `automation.set_room_temp_setpoint` and the legacy price offset helpers.
- [ ] Fix weak LK signal on Vardagsrum thermostat (RSSI -88 dBm, battery 28 %); it dropped out on 2026-10-10.
- [ ] Read the remaining documents: AirBox install manual, capacity diagrams, heat demand analysis (PDF/XLSX).

## Hot water (separate topic)

- [ ] Understand and decide how to control hot water (`warm_water_setpoint`, extra hot water, program).
- [ ] When it gets colder, hot water may take too much of the pump's time and starve house heating.
      Last 32 h at 11-15 C: valve on 11 % of the time. Check priority vs "VV-vaxeldrift" (HMC300 OM 5.7)
      and how much DHW blocks heating at low outdoor temperatures.

## Other

- [ ] `limit_compressor` / `limit_auxiliary_heat`: external inputs, not connected; revisit if wanted.
- [ ] Smart Grid / PV boost inputs (0-5 C boost): possible price or solar lever, see knowledge base.

## Price plan (see docs/superpowers/specs/2026-10-11-price-plan-design.md)

- [x] Plan 2: calibration, scoring and persistence (nightly refit, ScoreBoard, persisted mode and models).
- [ ] Watch `/api/model/health` `scores` in shadow mode; agree with Johan what 6 h and 24 h room prediction error is good enough before plan 4.
- [x] Room model replaced by a daily response model (gain about 0.4 C per C of flow, time constant about 4-5 days; see knowledge base). It does not predict the room temperature, it only gives the sensitivity.
- [ ] Verify the room response with a controlled test (plan 4): a few days at a lower or higher room setpoint, then compare the measured room change with the gain and time constant.
- [ ] Re-anchoring on every replan: check in shadow scores that the plan does not make the room sag day after day (floor 18 C protects; target 19.5 C is a soft cost).
- [ ] Plan 3: React frontend "Price and plan" page (copy bess-manager's Vite/React/Tailwind/Recharts setup, Docker Node stage, CI steps).
- [x] Plan 4: active mode built (executor with write guards, read-back, fallback; safety monitor; stages; savings). Not yet run against the real pump: switch with `POST /api/plan/mode {"mode":"active","confirm":true}` only after shadow scores look sane. Kill switch: `POST /api/plan/mode {"mode":"shadow"}` writes the base setpoint back.
- [ ] Watch `/api/plan/execution` and `/api/savings?days=14` in the first active week; check the compressor never stops and the aux heater stays quiet.
- [ ] (minor, plan 4) `PlanSettings` rejects `mode: active` in the config on purpose; the persisted state file is what keeps active across restarts.
- [ ] Plan 4 design: joint pump + LK decision (LK setpoint between target and ceiling per hour) and LK write guards.
- [ ] Re-check the 4 h sun lag with Open-Meteo's end-of-hour labels (may be about 3.5 h) and with stove evenings excluded.
- [ ] Existing `PriceOptimizer` parses `raw_today[].price` (SEK/MWh); retire it or fix it to use the SEK/kWh lists.
- [ ] **Adjust the heat curve manually on the pump (Johan does this; do it when the weather turns, before the cold).**
      Finding (see `documentation/ivt-heat-pump.md`, "Where the floor reaches its limit"): the living room needs a flow
      temperature of `42.4 - 0.98 * T_out` C (20 C indoors), but the pump curve at room setpoint 18 C gives
      `40.8 - 0.68 * T_out`. The pump is short by 1.6 C at 0 C, 2.5 C at -3 C and 4.5 C at -9.6 C, which matches the
      living room sitting 1-3 C under 19,5 C last winter. A steeper curve (about -1.0 C per C of outdoor temperature
      instead of -0.68) or a higher room setpoint offset fixes this down to about -2.9 C outdoors; below that the 45 C
      cap (LK's design flow) is the limit and the tile stove covers the rest (a few hours per season). Re-check against
      the kitchen/hall change below, which changes the heat balance. Do not exceed 45 C flow.
- [ ] Raise the winter delivery to the living room: pump curve is 1.6-4.5 C short of what the living room needs below 0 C (see knowledge base); decide between a higher room setpoint offset and a steeper curve, within the 45 C cap.
- [ ] Ask the parquet supplier for the warranty surface-temperature limit (45 C flow gives about 25-26 C at the surface, so 26 vs 27 C should not matter).
- [ ] **Kitchen and large hall renovation (planned).** The kitchen today has only a small radiator and is under-dimensioned;
      the plan is tile in the whole kitchen and the large hall (hall #1) with water-borne underfloor heating. Johan hopes
      heat will spread from the kitchen to the living room through the fairly open floor plan. When this is built:
      re-run the heat demand and LK dimensioning (kitchen 74,5 W/m2 at -9,6 C and tile gives a much higher output per m2
      than parquet, so the kitchen may no longer be the limiting room), re-fit the room model with the new circuits, add the
      new rooms' temperature sensors to the plan, and check whether the living-room deficit shrinks (it was 573 W at the
      design extreme). Until then the planner treats only the library and living room as comfort targets.
- [ ] (minor, from plan 2 review) Hold a lock around refit and model swap; copy the snapshot deque before iterating in `history()`.
- [ ] (minor) Failed state save after a successful refit reports "previous models kept"; `set_mode` changes the mode before saving.
- [ ] (minor) Room average jumps when one room sensor drops out (require all sensors); mask burn-in rows after long gaps.
- [ ] (minor) Hourly means are centred at h+0.5 but treated as start-of-hour values; Open-Meteo `temperature_2m` is instantaneous, not an hourly mean.
- [ ] (minor) Scoring mixes lead times by up to 1 h because plans simulate from the start of the current hour.
- [ ] (minor) Startup log prints the configured mode, not the effective (persisted) one; a damaged state file disables the plan service silently apart from one log line.
- [ ] (minor) Energy counter selection bias with 1 kWh resolution and 3 h forward-fill; small cleanups (dead `inputs_end` helper, comment wording, injected clock in `OpenMeteoClient.history`).

## Comfort sensor guardrails (2026-10-11)

- [ ] **Replace the Sonoff batteries** (vardagsrum `…045bf5`, bibliotek `…050818`; `…560e1` had 9 % in August) and check that they come back; if not, check the Sonoff bridge / eWeLink integration. The guard will dismiss its notifications by itself.
- [ ] Find out where the other two Sonoffs (`a4800560df`, `a4800560e1`) sit and add them to `sensor_guard` if they matter.
- [ ] LK Vardagsrum signal is weak (-70 to -88 dBm): consider a repeater; the comfort banner shows a yellow warning all the time until then.
- [ ] (minor, from review) Disagreement check is skipped in calm periods because forward-fill is only 2 h (should reach the silent-fault limit).
- [ ] (minor) A disagreeing sensor stays in the room average; only the non-reference sensor can be blamed for a disagreement.
- [ ] (minor) HA outage or restart reports every sensor as `unavailable` (false alarm if HA returns before the notification); `last_reported` could give tighter limits for Zigbee sensors.
- [ ] (minor) `health-banner.js` puts sensor names and messages into `innerHTML` unescaped.
- [ ] (minor) Banner fatigue: signal-only warnings keep the yellow banner on permanently.
- [ ] (minor) Clean-ups: dead `or EntityState(...)` in `SensorGuard.check`, `_RANK` duplicates `_SEVERITY_RANK`, `__all__` in `sensor_api.py`, use `dataclasses.replace`; cache the 7-day offsets for about an hour.
- [ ] (minor) Validate that the guard's reference sensors match `plan.room_temp_entities` (calibration uses the LK sensors only).

## Room model and planner (review 2026-10-11)

- [ ] (minor) `calibration.py`: `days` is built from hours with hour 0 while `daily_means` returns one more date: season labels are shifted by one day; add a length check in `DailyData`.
- [ ] (minor) The slab dimension of the DP is vestigial now (`tau_slab_h = 1`): remove it from the planner to cut the state space; `SLAB_INIT_FRACTION` is meaningless.
- [ ] (minor) Valve gating at LK 21 C may double-count what the daily gain already contains; shadow plans assume the valves stay open to 20.5 C while LK is at 19.5 C.
- [ ] (minor) `model_status` should say that the room model is a sensitivity estimate (e.g. `kind: "sensitivity"`), not a predictor.
- [ ] (minor) The leave-one-season-out CV pools seasons with one shared `b` and does not test the deployed median-of-seasons model.
- [ ] (minor) Tests identify rejection reasons by message text (`match=`); use distinct exception classes.
- [ ] (minor) An old-format `plan_state.json` stops the plan service at start; migrate or ignore it with a clear log line.
- [ ] Tune `setpoint_deviation_cost_sek_per_c_hour` (0.1 SEK per degree-hour, stands in for salon and guest house) and `comfort_cost` together with Johan once shadow plans can be compared with what the house did.
- [ ] **Plan 4: adaptive stages (trust region)** - see spec section 12: stage state, daily outcome measurement against a counterfactual baseline, widen/shrink rules, electric-heater term in the energy model.
- [x] Electric heater in the energy model: hinge on the flow target above 44 C, fitted from `aux_cons_heating` (live fit 2026-10-11: 0.20 kWh per C above 44 C, 921 hours). The planner now values the heater energy that lowering the setpoint saves in the cold. Not used by the savings backtest (`outcomes.py` still uses measured aux).
- [ ] The heater analysis has few hours for deep or long drops (below -3 C, more than a few hours): the first active stages must measure them rather than assume.


## Active-control review (2026-10-11) - fixed and deferred

Fixed with tests: stage-judgement failure no longer stops control (C1); one lock between `set_mode` and the control cycle, mode changes run in a worker thread (C2); leaving active restores the base first and keeps the mode (502) if that fails (C3); a hand change is noticed before the first write (I1); the base is written back at shutdown (I2); the compressor-stop check only counts while the setpoint deviates from the base, and a reported trouble starts the count over (I3); the heater check uses the real setpoint (I4); activating during a revert starts counting after it, widening needs a real saving (>= 0.5 SEK/day) (I5); trouble is judged on days with little data (I6, stages only); active is refused while `automation.set_room_temp_setpoint` is on (I7).

Deferred:
- [ ] Executor state (`_last_written`, pause) is not persisted; after a restart a hand change made during downtime is not seen until the first write.
- [ ] Startup with persisted `active` but a service that fails to build leaves the pump where it was (needs a base write at startup).
- [ ] Widening should also require realised saving >= 70 % of the predicted one and aux/room measured against the baseline (spec section 12); today fixed limits.
- [ ] Hot-water hours are dropped from aux and stopped-compressor counts in `outcomes.py`; trouble on those hours is invisible.
- [ ] Raise limit rounds up to the step (17.8 + 0.5 -> 18.5); fallback/restore jump straight to the base without a rate limit.
- [ ] Trouble notification is never dismissed; the HA `requests.Session` is shared by threads; savings days are UTC days.
- [ ] Counterfactual baseline scales each hour alone (recovery energy after a coast is charged to the baseline).
