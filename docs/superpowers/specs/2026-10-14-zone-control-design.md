# Zone control with the flow temperature as the lever - requirements and design (v2)

Status: approved and implemented 2026-10-14 (see section 14 for what was built and where it differs). Replaces the control parts of `2026-10-11-price-plan-design.md`
(sections on the LK thermostat lever, the observer and the stages). Measurements and manual quotes:
`documentation/ivt-heat-pump.md`. Open items: `documentation/TODO.md`.

## 1. Why a new design

On 2026-10-14 the plan pre-heated (pump setpoint 20, LK thermostats raised from 19.5 to 21 C). The
model predicted the ground-floor rooms to stay near 19.1-19.4 C; they went to 22.15 C (mean), 22.9 C in
the library, above the 21 C ceiling. Findings:

- The model has no input for the LK thermostat lever and does not model the sun (`k_sun = 0`).
- The "observer" compared the room with the *flat* prediction, so it read the plan's own effect as drift.
- The 21 C ceiling existed only inside the optimizer's prediction. Nothing acted on the *measured* room.
- The plan's freedom did not depend on how well the model had predicted.

The first spec said the lever is the pump only; the LK thermostat "boost" was added later without being
measured. This design returns to a small, physical model and puts a regulator on measured values.

## 2. Principles

1. A mathematical control model with a **small number of known-principle parameters** (first-order RC
   physics), each with physical bounds. Too many parameters make a loop hard to regulate.
2. Parameters are **adjusted at well-chosen times** (night for losses, sunny hours for the sun term), not
   continuously and not by opaque smoothing. Every change is logged and bounded per day.
3. **Measure and correct every 15 minutes**, as in an ordinary control loop: react to where the measured
   temperature is heading, whatever the cause (sun, stove, model error).
4. Hierarchy: **plan** (hourly, optimizes cost) -> **regulator** (15 min, protects limits) -> **estimator**
   (learns the model). The regulator never needs the plan to be right.

## 3. The system (facts the design rests on)

- One IVT AirX heat pump, outdoor-temperature controlled, no room sensor. The only influence is the heat
  curve; "room setpoint" `climate.room_temp_setpoint` is a **curve offset**: one step moves the flow target
  about 1.9 C (measured, stable 2022-2025). It is not a room temperature and is not treated as one.
- One flow temperature is shared by all zones: ground-floor floor heating (LK, 8 circuits, design flow 45 C,
  slow), the boutique/salon and the guest house (radiators with Netatmo / Tado thermostats).
- **Both floor systems (ground floor and upstairs) have their own mixing valves (shunts)**, which Johan has
  set to a **maximum of 40 C** for the flow to the floor loops, chosen so that the parquet surface stays at most
  26 or 27 C (the exact limit is unconfirmed; stated 2026-10-14; an earlier "36" was a misremembering). The floor therefore sees `min(T_flow, floor_mix_max_c)` with `floor_mix_max_c = 40`
  (a setting). For comparison, LK's revised calculation (LK023393) designs for a 45 C flow (parquet surface
  25.1-25.2 C at 20 C room temperature), and an external estimate puts a 27 C surface at about 40-42 C.
  With the flow fit (`T_flow = 40.6 - 0.71 T_out + 1.9 (rts - 18)`) the cap is reached below about +1 C
  outdoors at the base setpoint 18 and below about +6 C with the setpoint at 20: above those outdoor
  temperatures the pump setpoint heats the floor, below them only the valves do (section 8).
- Radiator calls for heat are available only from the smart radiators in the guest house and the salon
  (boutique); the kitchen and the upstairs floor have none. The upstairs floor is well insulated and holds
  its heat, so it needs no lower bound. The kitchen cannot be measured today and is out of scope.
- Measured: outdoor, flow (`heat_carrier_forwrd`, `radiator_forward`), return (`heat_carrier_return`),
  heating electricity (compressor + aux counters), per-circuit actuator state (`binary_sensor.lk_*_actuator_*`),
  room temperature per LK room. No flow meter; `circ_pump_speed` is not trusted (the register map is for the
  other pump type). The Sonoff sensors are currently unavailable, so each room has one temperature.
- Heating electricity is fitted on 8 500 hours: 0.224 kWh/h per setpoint step (90 % interval 0.21-0.24, stable
  over four periods). Flow minus return is used as a sanity check that heat is delivered, not as kW.

## 4. Levers and their roles

| Lever | Role |
|---|---|
| Pump curve offset (rts) | **The** control variable of the plan. Sets the flow temperature for all zones. |
| LK thermostat setpoint | A **per-room upper limit `U`**, held static. No boost. The thermostat closes a room's valves when the room reaches `U`. |
| Radiator thermostats | Not controlled. Their *calls for heat* are an input (a lower bound on the flow, section 8). |

## 5. Requirements

R1. One upper limit `U` per ground-floor room (default 22 C, set by Johan). The plan may store heat up to `U`
    when the economics say so (no separate "planned ceiling"); the LK thermostat is held at `U` so the room's
    valves close when it is reached. The plan leaves the margin by which the floor overshoots after the valves
    close (learned from measured events, section 7). If sun or a stove takes a room above `U` nothing more is
    done than keeping the valves closed and logging it.
R2. Valves are open by default. Lowering the flow in expensive hours must not close valves.
R3. Hard comfort floors as before (18 C day, 17.5 C night, 17 C at extreme prices) in the plan.
R4. The regulator runs every 15 minutes on measured values and works with the plan wrong or missing.
R5. The model has at most 3 parameters per room plus one shared floor lag, each with physical bounds; a fit
    outside the bounds is rejected and the previous parameters are kept.
R6. Every parameter change, every valve action and every refusal is logged with its reason.
R7. The plan's freedom (swing and credit for stored heat) depends on measured prediction skill on days the
    plan deviated from flat (section 9), not on the calendar.
R8. A prediction error larger than a set limit for a room switches the plan back to flat for that room's
    horizon and raises a notification.
R9. Whenever there is no sun (night, overcast) the model must predict a room's rise and cooling from the flow,
    the valve state and the outdoor temperature. These hours are where `a_i` and `b_i` are fitted and also the
    primary evidence of whether the model is right (hour-to-hour changes, not the day-to-day level that
    weather dominates).
R10. A room has one or more temperature sensors. The room temperature is the mean of its valid sensors
    (silent, flat-battery or disagreeing sensors are dropped by the existing sensor guard). Today only the LK
    sensor per room exists; the Sonoff sensors are added by configuration when their batteries are replaced.

## 6. Model

Per ground-floor room `i`, hourly:

```
dT_i/dt = a_i (T_out - T_i) + b_i v_i (T_flow - T_i) + c_i S + d_i
```

- `a_i` heat loss, `b_i` floor-to-room coupling, `c_i` sun gain; `v_i` is the **measured** fraction of open
  loops; `T_flow` is the (measured or planned) flow temperature; `S` is the forecast irradiation.
- The floor responds to the flow with one shared lag `tau` (fixed, 1 h) as today. `T_flow` in the equation is
  what reaches the floor: `min(T_pump_flow, T_mix)` with `T_mix = floor_mix_max_c` (a setting, 40 C; section 3). When the pump
  flow is above the cap a lower setpoint costs the floor nothing and saves electricity; it can only starve the
  radiators, which is what their calls for heat guard.
- `d_i` is one slowly varying disturbance per room (section 7). It replaces the observer.
- Bounds: `a, b, c >= 0`; `a`, `b` within a factor of 3 of the long-term fit; `c` within what the room's
  glazing can give.
- The flow temperature is `f(T_out, rts)` (fitted); the electricity is `g(T_out, rts)` (fitted).

## 7. Regulator (every 15 minutes)

The LK thermostat, held at `U`, already closes a room's valves when it reaches `U`. The regulator adds what the
thermostat cannot:

1. **Measure and correct.** Every 15 minutes: measure each room, compare with the model's prediction, update
   `d_i`, log the error.
2. **Learn the overshoot.** When a room reaches `U` and the valves close, record how far it rises before it
   turns (the floor keeps releasing heat, plus any sun). The plan uses `U - o_i` as the effective limit, with
   `o_i` bounded and updated as a slow average of recent events. On 2026-10-14 the library rose 1.9 C above
   the LK setpoint of 21 C, partly from sun.
3. **Verify the lever.** Check that the LK setpoints are what the design says (static `U`) and that valves
   actually close at `U`; log and notify if not (a failed write, a hand change).
4. **Anticipate (only if the overshoot is large).** If the projection with the floor state shows the room
   passing `U + o_i` after the valves close, close earlier. This is the only case where the regulator writes.

The regulator does not read the plan and works with the plan wrong or missing.

## 8. Decision: the flow temperature is shared with the radiators

Both levers answer to physics: the lowest efficient flow temperature is the lowest that satisfies the **most
demanding zone**; every zone with a lower need throttles with its own valve.

Options:

- **A. Flow only, valves always open.** Best efficiency for the floor alone, but the radiators in the boutique
  and guest house need a higher flow than the floor in cold weather. The flow then follows the radiators and
  the floor overheats with nothing to throttle it.
- **B. Valves regulate everything, flow high.** Robust, wasteful: a high flow with valves closing most of the
  time lowers the pump's COP.
- **C. Flow-led, valve-limited (recommended).** The plan chooses the flow (the only energy lever). The floor
  is protected from a high flow by its mixing valves, so what the radiators need cannot overheat or
  damage the floor; the LK setpoint is a static limit `U`, so a floor room that gets more than it needs is
  throttled by its own valve. A lower bound on the flow comes from the radiators' *calls for heat*
  (`hvac_action` of the smart radiators in the guest house and the salon): while they call and their rooms
  are below their setpoints, the plan may not lower the flow below the value that last satisfied them at this
  outdoor temperature (learned, one number per outdoor-temperature band). The kitchen and the upstairs floor
  have no signal and are not protected by this bound.

Consequence of C and the mixing valves: when the pump flow is above the mixing valves' cap, the floor's heat
input is fixed by the mixing valve and the valves are the only floor lever (open to charge, closed to hold);
the pump setpoint then only changes electricity and the radiators. Below the cap the pump
setpoint is the floor's lever as well.

## 9. Plan and authority

- The plan decides the flow offset per hour; the cost is price times modelled electricity. The end of the
  horizon is priced in energy (kWh per room degree from the models, at the horizon's average price) with credit
  for stored heat limited to 0.5 C; both are already built (`planner.py`).
- Authority is scaled by skill: on days the plan deviated from flat, record predicted vs measured room change
  and the cumulative deviation; the allowed swing (default +-2 C of curve offset) and the credit grow only
  when the last N such days were predicted within the limit (R7, R8). Until there is evidence the swing is
  small. This replaces the calendar stages that were removed.

## 10. Estimation schedule

| What | When | How |
|---|---|---|
| Room temperature, `d_i` | every 15 min | re-anchor on the measurement; `d_i` from the prediction error |
| `a_i`, `b_i` | nightly | from the night window (no sun, valve state known) |
| `c_i` | daily | from clear-sky hours with `a_i`, `b_i` fixed |
| Flow and electricity models | nightly | as today; bounded change per day |

## 11. Open questions

Resolved 2026-10-14: radiator calls exist for the guest house and the salon only (section 3); a room may be
closed at any flow temperature when it is over its limit; the upstairs floor needs no lower bound; the kitchen
is out of scope; the limit `U` is 22 C (in practice rarely reached: charging a house held around 19.5 C to 22 C
takes many consecutive hours); several sensors per room are averaged (R10).

1. Verify the 40 C cap against history (does the room's response to the flow stop growing above 40 C?).
2. The comfort floor (18 C) and the night values stay as configured.
3. What counts as "prediction within the limit" for R7/R8 (proposal: mean absolute error of the room change
   over 24 h on deviation days below 0.3 C, and of the hourly change in no-sun hours below 0.1 C).

## 12. Removed or replaced

- The LK "boost" (`lk_setpoint_c` as a plan-driven lever) -> static `U`.
- The drift observer (`observer.py`) -> `d_i` in the regulator.
- The room response fitted from daily windows (`daily_room.py`) -> `a_i`, `b_i`, `c_i` per room from the
  schedule above.
- Calendar stages -> evidence-based authority (R7).

- The planner's `max_flow_c` (45), which kept the plan from asking for a flow the floor could not take -> the
  floor is protected by the shunts at 40 C; `max_flow_c` stays only as a sanity bound on the pump flow.

## 13. Verification

- Shadow mode first: the regulator computes its actions and logs them without writing.
- Replay: the plan and regulator run against the stored history of 2026-10-13/14 and must reproduce
  "the room is held near 21 C instead of reaching 22.9 C".
- Acceptance: over two weeks, no room above `U + o_i` by more than 0.3 C for more than 30 minutes; plan cost vs
  flat reported with the end-of-horizon payback counted; prediction skill on deviation days published on the
  Models page.

## 14. Implementation status (2026-10-14)

Plan: `docs/superpowers/plans/2026-10-14-zone-control.md`. Built:

- Settings: `lk_limit_c` (static LK limit), `lk_base_c` (where the thermostats stand when Theria does not
  control them), `floor_mix_max_c` (40), `ceiling_c` (22), `room_controls`, `regulator_active`, `radiator_entities`.
- Model: the floor sees `min(flow, floor_mix_max_c)`; the sun is a swing on the slow room level
  (`sun_swing_c_per_wm2`, lag fitted 1-6 h) and not a heating rate (a rate would pile up in a slow model);
  fitted nightly from a year of heating hours, bounded step per day. On Johan's data: 0.0067 C per W/m2, lag
  4 h, error in the sunny hours 0.90 -> 0.60 C.
- `LKExecutor`: static limit, per-thermostat holds from the regulator, 1 h pause on a hand change.
- `disturbance.py` replaces the observer: one-step innovation from measured flow, open loops, outdoor and sun.
- `regulator.py`: per room, mean of valid sensors, learned rise after the valves close, closes earlier by it.
  `regulator_active: false` by default: it measures and logs "would: ..." until its decisions have been read.
- `skill.py`: authority in [0.5, 1] from the systematic error (6 h block means) in dark hours; the plan's
  setpoint range around the base is scaled by it. Evidence is persisted.
- Radiator calls: no cuts below the base while a radiator reports `hvac_action == heating`; unreadable
  entities are ignored but reported (all were `unavailable` when this was built).
- UI: Rooms panel (Control), sun swing and disturbance (Models), 15-minute prices (Overview).

Differences from the design above:

1. One room-level state in the planner, per-room logic in the regulator.
2. `a_i`, `b_i` stay on the daily response and night drift; the nightly hourly fit needs nights with the
   actuator history, which exists only since 2026-10-13. TODO.
3. The radiator bound is the simple rule (no cuts while calling), not a learned flow per outdoor band. TODO.
4. Authority uses all dark hours, not only deviation days; deviation-day evidence needs more days of data.
5. The 15-minute innovations are mostly sensor quantisation (0.1 C), so skill is judged on 6-hour means.
6. A latent planner bug was found and fixed: a setpoint range narrower than the rise limit broke the DP.

### UI (2026-10-14)

- Charts and tables show whole local days (00:00-24:00): the chosen day and the next when it has data. Plan, flat
  baseline and measurement are in the same chart; for hours that have passed the plan is the one that was applied
  (`plan_record.py`, one line per hour in `/data/plan_record.jsonl`, so history before 2026-10-14 has no plan).
- Saving per hour against the flat setpoint replaces the running cost chart; the schedule strip (pre-heat, cut, hold)
  follows the BESS Manager's mode timeline.
- Settings page: the plan settings are editable (`settings_manager.py`, overrides in `/data/plan_overrides.json`,
  applied live, resettable). Models page: every parameter and weight (`model_catalog.py`), with source and meaning.
- The tuning advice was rewritten for the shunts (floor limit) and the authority (setpoint limits are not judged
  while the plan has half the range).
