# TODO - things to look at later

## First floor heating (current focus)

- [x] Quantify `room_temp_setpoint` effect on `heating_setpoint`: ~1.9 C per C, see knowledge base.
- [ ] Measure how fast first-floor temperatures respond to a flow-temperature change (concrete lag).
- [ ] Compare observed first-floor temperatures with the setpoint/outdoor history across seasons to size the
      lever Theria should use (curve shift vs LK setpoint).
- [x] LK actuator state is exposed by Johan's `ha-lksystems` fork (PR angoyd/ha-lksystems#145) as
      `binary_sensor.lk_<room>_actuator_<n>` (device_class opening, on = loop open): Vardagsrum 7, 9, 11;
      Bibliotek 4, 6, 8, 10, 12. LK thermostat writes work again (fixed by Johan 2026-10-13).
- [x] Day view: "Open floor loops" chart per room (`plan.room_actuators`, `loops_open`/`loops_total` in
      `/api/plan/day`). Needs the actuator entities in the HA InfluxDB integration (they are not stored yet, so the
      chart is empty until they are included).
- [ ] Use the loops further: show whether the LK boost really opens the floor loops, and feed the room model with
      valve state once there is history.
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
- [ ] Watch `/api/plan/execution` and `/api/savings?days=14` in the first active week; check that the aux heater stays quiet.
- [ ] (minor, plan 4) `PlanSettings` rejects `mode: active` in the config on purpose; the persisted state file is what keeps active across restarts.
- [x] LK lever (decided with Johan 2026-10-11): thermostats go to `lk_setpoint_c` (21 C) only while the pump setpoint is actually above its base, otherwise `lk_base_c` (19.5 C); never lowered below the base; hand changes pause it 1 h (was 6 h until 2026-10-13: Johan had not agreed to 6 h); thermostats set below the base by hand are left alone; restored with the pump setpoint. Lowering LK below 19.5 to coast is NOT done (needs the buffer-tank/defrost check first).
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
- [x] Executor state (`_last_written`, pause) is persisted in the state file, so a hand change made during downtime is seen after a restart.
- [ ] Startup with persisted `active` but a service that fails to build leaves the pump where it was (needs a base write at startup).
- [ ] Widening should also require realised saving >= 70 % of the predicted one and aux/room measured against the baseline (spec section 12); today fixed limits.
- [x] Hours that are mostly heating (hot water < 50 %) are watched for aux use and a stopped compressor, although they are left out of the cost.
- [x] Raise limit rounds down to the step. [ ] Fallback/restore still jump straight to the base without a rate limit.
- [ ] Trouble notification is never dismissed; the HA `requests.Session` is shared by threads; savings days are UTC days.
- [ ] Counterfactual baseline scales each hour alone (recovery energy after a coast is charged to the baseline).

## Before the first active night (2026-10-11)

- `GET /api/plan/preflight` (and the Control tab) lists what active mode depends on; `POST /api/plan/mode` refuses `active` while a check fails.
- `GET /api/plan/log` and the Control tab list every setpoint write, trouble, stage and mode change since the add-on started (in memory: lost at restart).
- A persistent notification "Theria: yesterday" summarises the previous day (saving, room min/mean, heater kWh, compressor stops) when the day is judged (first cycle after 00:00 UTC).
- [x] Persist the action log across restarts (`plan_events.jsonl`).
- [ ] Startup in persisted `active` mode with a service that cannot be built leaves the pump where it was.

## Self-learning (2026-10-11)

Learned continuously (see the Models tab and `GET /api/plan/parameters`, which keeps one record per refit and per change, persisted in `plan_parameters.jsonl`):
- pump flow curve, energy per setpoint, electric heater coefficient, room response (nightly refit)
- **night cooling** (`night_drift.py`): room change over the night window by outdoor temperature, steady-setpoint nights only, half-life 150 days. Live fit 2026-10-11: -0.48 C per night, almost independent of outdoor temperature (slope 0.01), 302 nights, error 0.35 C. Few cold nights, so the cold end is uncertain; the table of recent nights (predicted vs measured) shows how it follows.
- **night margin** (`morning.py`): raised 0.25 C when the room at 06:00 local was more than 0.3 C under the target, lowered 0.1 C after 3 comfortable mornings; lifts the night target and floor (cap: target - night_target). Learns only in active mode.
- stage (trust region).
Defaults: `night_target_c` 18.5, `night_floor_c` 17.5 (careful until cold nights are measured).

- [x] Estimated lead time before comfort starts: `comfort_outlook` in every plan (room at 06:00, deficit, extra warm-up rate, hours of maximum heating the deficit would need), shown on the Overview. Live 2026-10-11: 19.47 C, 0.03 C short, about 2 h. - [ ] Done below (warm-up scale).
- [x] Residuals checked on 301 real nights: no weekday pattern, a weak evening-warmth effect (not worth a term), but the **room level at the start of the night matters** (a warmer room loses more): added. Fit error 0.35 -> 0.31 C; the outdoor effect then shows: about -1.3 C at -10 C and -0.3 C at +10 C for a 19.5 C room.
- [x] Alert (notification, dismissed again when fine) if the recent nights are missed by more than max(0.4 C, 1.8 x the fit error).

## Warm-up speed (2026-10-11)

`warmup.py` measures how the room answers setpoint changes: per 24 h window the room change is regressed on the change of the mean setpoint (raising and lowering separately), with weather change, outdoor temperature and room level as controls, recent windows weighted more (half-life 365 days), and compared like for like with what the model's own simulated room gives for the same setpoints. The warm-up scale is applied to `k_slab` in the planner only when it differs from 1 by more than 2 standard errors (clipped 0.5-1.5).

Real data (4 years, 635 days, 84 with a setpoint change): warming scale 0.04 +- 1.6 (not identifiable; 1.0 is used), cooling scale 2.2 +- 1.2 (about 0.25 C per C of setpoint over 24 h against 0.09 for the model).
- Why warming is not seen: the LK thermostats stood at 19.5 C, so a higher pump setpoint did nothing once the room was there. With the room below 19.3 C (valves open) the warming looked like 0.10-0.14 +- 0.08 per C, in line with the model.
- Why cooling may be overstated: the old manual cuts were mostly made when the room was already warm (sun, stove), so part of the cooling is not caused by the cut.
- [ ] Re-read the Models tab after the first active nights: the plan's own setpoint changes (and the LK boost) are clean experiments and will show the real warm-up and cooling speed.
- [ ] If cooling stays 2x the model: give the room model an asymmetric response (cooling faster than warming) so that cuts are costed correctly.

## Found in the first active day (2026-10-12)

- [x] Keep the stage judgement's reason (done): `last_stage_note` was not persisted and the action log was lost at the update, so the revert to the base until 2026-10-15 cannot be explained. Persist the note and show it on the Control tab. The day data (savings) shows no trouble (no compressor stop, no heater, room minimum >= 19.17 C against the 19.0 C limit), so check what `_judge_stage` actually saw.
- [ ] Plan at 16 C because the room is above the ceiling (21.35 C, extreme price) while the stage holds the pump at the base: the plan table should say "held at base by the stage" so it does not look like a 16 C setpoint.
- [ ] Find out why the room is 21.35 C (stove, sun) and whether the ceiling case needs its own handling.
- [x] Tab links broke in the HA frame (the `<base>` made `#control` point at another document); `tabHref` fixes it.
- [ ] "Tuning" tab: per setting (target, floor, night target and floor, rts_max, stage readiness) the current value, what the data says, and a suggestion with the reason; never applied automatically.
- [x] Charts always show whole local days (`GET /api/plan/day`, Today/Tomorrow): measured setpoint and room for the hours that have passed, plan for the rest, "now" marker. The Plan tab table still starts at the current hour.
- [ ] Day charts: one hour (the current one) has neither a measured nor a planned point on the line, so the lines have a gap at "now"; the Models/Plan tabs are not on whole days yet.
- [x] The plan starts from the room's usual level (`room_level.py`: the lower of the measured room and the median of the last 24 h), so an afternoon of sun (room 22.7 C, no stove) is not planned as if the room stayed that warm for days. Check in the first sunny days that the morning room matches the outlook; the sun itself is still not modelled (`k_sun` = 0).
- [x] The Plan tab has charts (flow temperature against the 45 C cap, energy per hour, running cost plan against flat) above the table.

## Safety rules removed (2026-10-12)

- [x] The compressor-stop guard is gone (in the live and the daily check). The compressor stops for 20-70 minutes every 1-2 hours at mild weather whatever the plan does, so every stop was a false alarm. The stop counter is no longer measured either; the preflight "compressor" check is removed too.
- [x] `revert_days` and the pause at the base after trouble are gone. Trouble (electric heater while raised, room more than 0.5 C under the target in a day) narrows the stage one step and starts the judging over; there is no pause.
- [ ] Without a pause, a heater that runs every time the setpoint is raised makes the plan raise and restore again about once an hour (one raise per 55 minutes is the only brake). Watch the action log for that.
- [x] The reason of the stage judgement and the last trouble are persisted and shown on the Control tab.

## Control loop (2026-10-12)

- [x] Level 1, fast: `observer.py` compares the room's level now with what the flat-setpoint prediction of a plan 2-4 h old expected and adds the smoothed shortfall (gain 0.3, half-life 4 h, cap 0.3 C/h) to the planner's drift, fading with the horizon. Not persisted. The room level (not the sunny reading) is also what the drift anchoring and the night model use now (that was a bug in the first room-level change).
- [x] Level 2, daily (first part): `trim.py` limits how far a refit may move what the planner uses per day: night intercept 0.10 C, slope 0.02, level 0.05, warm-up scale 0.10; every cut is named in the parameter history note. The morning margin and the stage already move in small steps.
- [x] Level 2, second part: the room response (gain 0.05, rate 0.03 per day) is trimmed the same way, with the hourly parameters rebuilt from the limited response; the night fit now also uses nights the plan moved the setpoint in (up to 2 C), correcting the change for the setpoint's small effect (`SETPOINT_EFFECT_C_PER_C_H`).
- [x] Level 3: the Tuning tab (`tuning.py`, `GET /api/plan/tuning`): floor, flow cap, setpoint limits, stage readiness and the night model, each with the current value, what the days show and a value to consider; nothing is applied. More rules (night target, `lk_setpoint_c`, the cost weights) need data the days do not give yet.
- [x] Found by the tab: the outcome `coverage` counted hot-water hours as missing data (0.13-0.42 on real days), so the stage could never widen (it needs coverage >= 0.5). It is now the share of the non-hot-water hours with complete data.
- [ ] Check in a few days that the observer does not make the plan flutter (the parameter history records `observer_drift_c_per_h` at every refit).
- [x] Read-back waits for the pump: Husdata confirms a setpoint 3-6 s after the MQTT write ("HP Set confirmed!"), so the immediate read-back saw the old value, reported "failed" and the unchanged `_last_written` then looked like a hand change (6 h pause). `confirm.read_until` reads up to 10 times, 2 s apart, for the pump and the LK thermostats.

## Stages removed (2026-10-13)

- [x] The stage (trust region, +/-1 C then +/-2 C) and its daily judgement are gone: Johan prefers to watch the
  result. The executor keeps the setpoint inside `rts_min_c`..`rts_max_c`, raises at most `max_rts_step_per_hour_c` per
  55 minutes and reads every write back. Trouble (electric heater running while the setpoint is raised) is logged and
  notified, nothing is restored. The daily review still learns the morning margin and sends yesterday's summary.
- [x] A hand change pauses the pump and LK writes for one hour; stopping and starting active mode ends the pause.
- [ ] Watch the electric heater (`add_heat_status`, aux energy) now that nothing narrows the plan after trouble; if it
  runs when the plan raises, lower `rts_max_c` or `boost_flow_limit_c` (see the Tuning tab).
- [x] `max_rts_step_per_hour_c` is 2 (was 1): the setpoint range 16-20 can be crossed in one step. It was an assumption, not a measurement; watch the electric heater when the plan ramps up.
