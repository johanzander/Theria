# Price-driven heating plan for the ground floor - design

Status: draft for review (2026-10-11). Knowledge base with the underlying measurements:
`documentation/ivt-heat-pump.md`. Open items: `documentation/TODO.md`.

## 1. Goal

Reduce the cost of heating the ground floor (Bibliotek and Vardagsrum, floor heating with LK control) by
using the building's thermal mass as a heat store: heat less when electricity is expensive, pre-heat when it
is cheap, and never go below a hard comfort floor. The pump is what consumes electricity, so the control lever
is the pump's room setpoint (`climate.room_temp_setpoint`, H60), not the LK thermostats.

Success criteria:

1. Over a heating season the modelled plan costs less than a flat 18 C room setpoint baseline, shown in SEK on
   the plan page and verified against measured pump energy.
2. Ground-floor temperatures never go below the hard floor (18 C, 17 C at extreme prices) and rarely above
   the ceiling (21 C).
3. Johan can see, for every hour, what the model predicted, what happened and why the plan chose that
   setpoint ("full insight into what the model does and thinks").

## 2. Scope

In scope: heating only; ground floor as the comfort target; the pump's room setpoint as the lever; shadow mode
first, then controlled tests, then active control within a narrow band, widening stepwise.

Out of scope for this spec (tracked in TODO): hot water, boutique and guest house as comfort targets (they share
the pump and will see the changed flow temperature but are not modelled), tile-stove detection, buffer tank,
window upgrade effects, Smart Grid / PV boost inputs.

## 3. What the measurements say (inputs to the design)

All numbers and caveats are in the knowledge base. The ones the design depends on:

- Pump flow target follows outdoor temperature (3-6 h smoothing) on a manually set curve:
  about `40.8 - 0.68 * T_out` since Feb 2026.
- One degree of room setpoint moves the flow target by about 1.9 C within an hour (median of 227 steps,
  stable across 2022-2025).
- The compressor never stops in heating mode; it modulates. Heating electricity is roughly 7 % lower per degree
  lower room setpoint (rough, unit of counter uncertain). Savings come from modulation, not on/off.
- Rooms follow sun with about 4 h delay; Vardagsrum responds more than Bibliotek. Stove evenings confound this.
- Winter 2025/26 the rooms were 1-2.7 C below the LK setpoint, so the baseline itself can be under target.
- Price spread, not level, is what can be exploited: median daily 4-hour-block spread in heating season is
  0.83 SEK/kWh, 90th percentile 1.9; this autumn volatility is higher than in previous years.

## 4. Architecture

New backend components under `core/theria/` (names indicative), each with one purpose:

| Unit | Purpose | Depends on |
|---|---|---|
| `price_service` | Fetch Nordpool prices from HA (15-minute resolution, today + tomorrow when available), keep history, classify extremes | `ha_client`, existing `PriceOptimizer` classification where reusable |
| `weather_service` | Open-Meteo forecast and archive: shortwave and direct radiation, cloud cover, temperature, for HA's location | HTTP, HA location |
| `pump_model` | Flow target and electricity as functions of outdoor temperature and room setpoint; self-calibrating from InfluxDB/HA | `ha_client`, InfluxDB history |
| `zone_thermal_model` | Two-state (slab, room) model per ground-floor room with sun and disturbance term; refit nightly | LK sensors/setpoints, pump_model, weather_service |
| `planner` | Choose a room setpoint per period for the next 24-36 h minimising price x energy + comfort cost, subject to floor and ceiling | models above |
| `plan_executor` | Mode handling (off / shadow / active), write guards, read-back, fallback, audit log | `ha_client`, planner |
| `plan_store` | Persist plans, predictions, actuals, reasons and parameter history for the UI and for scoring the model | existing `history.py` pattern |

Existing `PriceOptimizer` stays until the new path has replaced it; the new planner must not depend on its
per-hour adjustment API. `backend/api.py` gains plan endpoints; `config.yaml` gains a `plan` section.

Frontend: a new React app in `frontend/` (see section 7), built like bess-manager (Vite, React, TypeScript,
Tailwind, Recharts, Vitest). The existing static pages in `backend/static/` keep working unchanged and are not
ported in this scope; navigation links between the two. The Dockerfile gains a Node build stage (as in
bess-manager) and CI gains frontend lint, type-check and test steps.

Data sources: Home Assistant (`sensor.outdoor`, `sensor.heating_setpoint`, `sensor.radiator_forward`,
`climate.room_temp_setpoint`, LK temperatures and setpoints, Nordpool sensors), InfluxDB (history; same schema
as home-analytics: measurement = entity id, field `value`, climate attributes as fields), Open-Meteo.
Energy numbers reuse the spike/outlier cleaning approach from `home-analytics` (`_remove_outliers`,
gap-spike repair, monotonic counters).

## 5. Model

### 5.1 Pump

- Flow target: `flow_target = curve(T_out_smooth) + 1.9 * (RTS - RTS_ref)`, where `curve` is a linear
  fit recalibrated weekly (the curve is changed manually on the pump and not logged) and `RTS_ref` is the room
  setpoint during the calibration window.
- Electricity: `E_kwh_per_h = a - b * T_out + c * RTS` fitted on recent heating hours, initial `c` about 7 % of
  mean per degree. Reports its own residual error. Hot-water hours are excluded from fitting; the planner
  treats hot-water time as lost heating capacity but does not control it.

### 5.2 Room (changed after plan 2)

A hourly two-state room model could not be fitted so that it beat "no change" in free-running tests, so the room is modelled as a
**sensitivity** only: the daily response of the room temperature to the flow temperature (gain about 0.4 C per C, time
constant about 4-5 days, fitted per season on daily means, see `core/theria/plan/daily_room.py` and the knowledge base).
Every plan starts from the measured room temperature and anchors the baseline there; the model gives how much the plan's
deviation from the baseline moves the room. Predicting the room temperature itself is out of scope.

### 5.3 Planner

- Horizon: now to the end of the known price window (24-36 h). Decisions are hourly; the cost of an hour
  uses the mean of its four 15-minute prices (amended during planning: hourly steps keep the DP state grid
  fine enough and match the rate limit). Checked against price history since Oct 2025: quarter-hour
  resolution is worth at most about 1.3 % of the daily mean price for an ideal load; revisit in plan 2.
- Decision: room setpoint per period in 13-22 C, 0.5 C steps, rate-limited (default max 1.5 C per hour).
- Objective: `sum(price_t * E_t) + comfort_cost * sum(max(0, target - T_room_t))`.
  Default `comfort_cost = 1 SEK per C-hour` (assumption, adjustable on the page).
- Hard flow-temperature cap: the pump's flow target may not exceed `max_flow_c` = 45 C, the flow temperature
  LK designed the floor for; the floor is not used above its design (parquet). Setpoints that would exceed it
  are not allowed in the plan; where it binds the reason says `flow_cap`. The outdoor temperatures at which
  rooms reach the cap are in `documentation/ivt-heat-pump.md`.
- Constraints: `T_room_t >= floor`, `T_room_t <= ceiling`. Floor 18 C; 17 C when the period's price is
  "extreme" (default: top 5 % of the last 14 days and at least 3 SEK/kWh; configurable, to be confirmed by
  Johan). Target 19.5 C, ceiling 21 C (assumptions, to be confirmed).
- Solver: dynamic programming over the discretised setpoint and a coarse temperature state, so the search is
  exhaustive and every decision can be explained. No external solver dependency.
- Output per period: chosen setpoint, predicted room temperatures, predicted flow target and energy, cost,
  baseline cost (flat 18 C) and a reason record (price rank, marginal value of +/-1 C, comfort cost, binding
  constraint).

Replan every 15 minutes and whenever new prices arrive (normally early afternoon for tomorrow).

**LK is a second control variable (decided with Johan after planning).** The LK valves close as the room
reaches the LK setpoint (19.5 C today), so with LK at 19.5 C the floor cannot store heat above 19.5 C and
pre-heating does nothing; but simply leaving LK at the ceiling (21 C) would let the sun and stored heat
overheat the rooms (spill heat). In active mode the planner therefore chooses both the pump room setpoint and
the LK setpoint for each hour (LK between target and ceiling): LK high while pre-heating, back to the target
when the floor should stop taking heat. Plan 1 (shadow) models LK as one fixed input, `lk_setpoint_c`
(default 21 C), which makes its predictions "pre-heating allowed"; the joint decision, its write guards and
the LK writes belong to plan 4.

## 6. Modes, execution and safety

Mode is stored in config and changeable from the page; default `off`.

| Mode | Behaviour |
|---|---|
| `off` | No planning, no writes |
| `shadow` | Plans and logs, writes nothing; predictions are scored against reality |
| `active` | Writes `climate.room_temp_setpoint` per the plan |

Rollout: shadow for several days; then a small number of explicitly approved test steps (about +/-2 C for a
few hours on mild days) to identify the flow-to-room response; then `active` within a narrow band (for example
17-19 C) widened as prediction error allows. Each change of mode or band is made by Johan.

Write guards in `active`:

- Clamp to the configured intervals (default 15-22 C) and to the maximum rate of change.
- Read back after every write and log mismatches; raise a health event on repeated mismatch.
- Fall back to the base value (18 C) and raise a health event if prices are missing, key sensors are stale
  (using existing health monitoring), model error exceeds a limit, or Theria cannot reach HA.
- Kill switch on the page that sets mode to `off` and restores the base value.
- Verify at startup and periodically that `automation.set_room_temp_setpoint` remains off; refuse `active`
  if it is on.
- Only `climate.room_temp_setpoint` is written. No LK, hot water or curve parameter is touched.

## 7. API and UI

Endpoints (indicative): `GET /api/plan` (current and next-day plan with predictions and reasons),
`GET /api/plan/history` (past plans against actuals), `GET /api/model/health` (parameters, errors, last refit),
`POST /api/plan/mode` (mode and limits), plus reuse of existing price and heat pump endpoints.

New React page "Price and plan" (route in the new `frontend/` app, served by the Theria backend next to the
static pages; must work under Home Assistant ingress, i.e. relative asset base and API paths as bess-manager
does; reuse the health banner's data from `/api/dashboard-health-summary`). Chosen over extending the static
Chart.js pages because the page is interactive (mode and limit controls, selected-period "why" panel, linked
charts) and bess-manager already provides a tested stack and conventions to copy:

1. Price and plan, today and tomorrow: price per period, planned room setpoint as a step line, baseline,
   floor / target / ceiling bands.
2. Room temperatures: predicted vs actual for Bibliotek and Vardagsrum with uncertainty band, sun and outdoor
   temperature underlay.
3. Pump and cost: predicted vs actual flow target and energy; cumulative SEK for plan vs baseline.
4. "Why" panel for a selected period with the reason record.
5. Model health: parameter history, prediction error, last refit, prior-vs-learned flags.
6. Controls: mode, comfort cost, floor, target, ceiling, kill switch.

## 8. Testing

- Unit tests for the planner on synthetic price curves and fixed model parameters: flat-price days stay within the rate limit
  and setpoint range (they may still deviate from the baseline to close the comfort gap); large spread produces pre-heat then sag; the floor and ceiling are never violated; extreme-price
  rule lowers the floor only in extreme periods.
- Model tests with recorded data from InfluxDB: pump flow/energy fit reproduces the measured 1.9 C/C and about
  7 % per C within tolerance; room model prediction error reported on a held-out period.
- Executor tests with a fake HA client: shadow never writes; active clamps, rate-limits, reads back, falls
  back on missing prices or stale sensors; refuses to start with the old automation on.
- API tests for response shapes. Frontend: Vitest component tests for the controls, the reason panel and
  empty/error states; type-check and lint in CI; UI checked manually against a mock HA (`mock-run.sh` extended
  with price and plan data).
- Behaviour-focused, per `CODING_STANDARDS.md`; no string matching on exception messages.

## 9. Assumptions to confirm

1. Target 19.5 C, ceiling 21 C, floor 18 C (17 C at extreme prices).
2. Definition of "extreme price" (top 5 % of last 14 days and at least 3 SEK/kWh).
3. Default comfort cost 1 SEK per C-hour.
4. Prices from `sensor.nordpool_kwh_se4_sek_2_10_025` are what Johan pays at the margin; confirm what the sensor
   includes (VAT, supplements) because VAT scales the spread.
5. Room setpoint range 15-22 C and 1.5 C per hour maximum change.
6. `max_flow_c = 45` C is LK's design flow temperature, confirmed by Johan (earlier 50 C draft removed).
7. In active mode Theria also writes the LK setpoints (between target and ceiling); write guards for LK are part of plan 4.

## 10. Delivery in four plans

1. Backend in shadow mode (models on priors, planner, API): `docs/superpowers/plans/2026-10-11-price-plan-backend-shadow.md`.
2. Calibration and scoring (InfluxDB history, nightly fits, prediction error, persisted snapshots and mode).
3. React frontend (the page in section 7).
4. Active mode (executor, joint pump + LK decision, guards, controlled test steps).

## 11. Risks

- Flow-to-room coupling is unmeasured until test steps are done; early plans carry high uncertainty, shown
  explicitly.
- Stove, sun and window changes are disturbances the model sees only as residuals.
- Adding a Node toolchain to Theria (Dockerfile stage, CI, ingress base path) is extra setup and a second UI
  stack until the old static pages are ported; mitigated by copying bess-manager's working configuration.
- The room setpoint affects the whole house (boutique, guest house); the first step only optimises for the
  ground floor and must be watched for comfort complaints elsewhere.
- Rough energy-vs-setpoint relation (counter units uncertain) may overstate savings; the page reports measured
  against predicted energy so the claim is checkable.
- Winter: if the system cannot hold 19.5 C even at baseline, the plan has little room to sag; the hot-water /
  heating priority question remains open (TODO).

## 12. Adaptive operation (design for plan 4, decided with Johan 2026-10-11)

What adapts today: the nightly refit re-estimates the pump curve, the heating-energy model and the room response from all
history, so once the plan runs it also learns from the days it controlled. What does NOT adapt: the size of the deviations
the plan is allowed, and nothing compares what the plan predicted with what happened to decide how far to go.
Plan 4 adds a measured, stepwise widening ("trust region"):

1. **Stages (dropped 2026-10-13: the plan now acts inside the full setpoint range from day one and the result is watched; items 1, 3 and 4 below no longer apply).** Shadow (now) -> active within +/- 1 C of the base setpoint (rise at most 0.5 C per hour) -> +/- 2 C (rise at
   most 1 C per hour, the manual range) -> only then more. The stage is stored in the state file and changed by the executor, not by hand.
2. **Daily measurements, per day of active control:** realised heating electricity (compressor and aux counters) against the
   counterfactual baseline (the energy model at the base setpoint for the actual weather); aux energy and hours; the room's minimum and mean against target; the scoreboard errors; the room's measured change against the response
   model.
3. **Widen** one stage after N (for example 5) days when the realised saving is at least about 70 % of the predicted one, aux
   energy is not above the baseline level, there was no compressor stop, and the room stayed within 0.3 C of the baseline.
4. **Shrink** (one stage down, judging starts over) when aux energy rises, the room
   falls more than 0.5 C below the target, or the realised saving is negative over a week.
5. **Learn.** The energy model gets a term for the electric heater; the energy-per-setpoint slope and the room response are
   refitted with the active days weighted up; a model that predicted the day badly lowers the stage as above.
6. Every step is logged with its reason and shown on the plan page, so Johan can see what the model assumed, what happened,
   and why it changed course.

