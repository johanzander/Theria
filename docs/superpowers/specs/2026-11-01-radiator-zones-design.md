# Salon and guest house: price-driven radiator setpoints - design

Status: draft for review (2026-11-01). Builds on `2026-10-14-zone-control-design.md` (the pump setpoint is the
plan's lever; radiators are only a read-only block today). Knowledge base: `documentation/ivt-heat-pump.md`,
open items `documentation/TODO.md`.

## 1. Intent

Johan wants the two radiator zones to take part in the price saving the way the ground floor already does:
less heat in the dearest hours, a little extra in the cheap ones before them. The zones are the salon (a shop,
35 m2, Netatmo thermostats) and the guest house (60 m2, Tado thermostats). Both sit on the pump's full flow,
with no mixing valve.

There is a second reason. Today the plan may not lower the pump setpoint below its base while any radiator
reports `hvac_action == heating` (`radiator_entities`, `service.py` `_read_radiators`, held for one hour).
Right now both salon thermostats call at 100 %, so they block every flow cut in the whole house. Controlling the
radiators is therefore a precondition for cutting deeper in dear hours, not only an extra saving.

Success: (a) the salon is at the right temperature while open, (b) the guest house stays within its band,
(c) the radiators stop blocking the pump cuts, and Theria reports in SEK what the radiator control saved (or
would have saved, in shadow mode). Anti-goal: saving a few öre at the price of a cold shop. Johan said that
saving too little is the more acceptable error, but a salon below its target during opening hours is still a
failure.

## 2. Scope

In: schedule and settings in the Theria UI; a price-driven setpoint plan for both zones; a new executor that
writes the Netatmo and Tado setpoints; the link to the pump plan (the block is lifted while the radiators are
under Theria's control and their rooms are at or above their lower bound).

Out: the ground-floor floor heating (unchanged), hot water, the kitchen's own planned underfloor heating,
changes to the pump plan's optimizer beyond the radiator block. The first delivery runs in **shadow mode**
only (section 8).

## 3. The zones as measured today (2026-11-01 18:00)

Salon: `climate.vantsidan` (20.5 C, target 21, heating, power request 100 %), `climate.klippsidan` (20.0,
target 21, heating, 100 %). Both are Netatmo, modes `auto`/`heat`, step 0.5, range 7-30 C, and sit on a
schedule (`select.salong` = "Mitt schema"). Other temperature points in the salon: the Verisure water alarm
on the floor, `sensor.schamponeringstolen_temperature` (19.7 C), and the Verisure smoke detector in the
ceiling, most likely `sensor.butik_temperature_2` (19.1 C; to be confirmed with Johan). They report every few
minutes to a few hours (timestamps 17:01-17:50 for four sensors). `switch.vattenavstangning_salongen` is a
water shut-off valve and is not touched.

Guest house: `climate.bedroom` (18.0 target 18.0), `hallway` (20.9 / 19.5), `kitchen` (20.9 / 20.0),
`tv_room_gh` (20.0 / 19.5), `storage_room` (18.6 / 17.0). Tado, modes `off`/`auto`/`heat`, step 0.5, range
5-25 C, `hvac_action` available.

The salon has a thermal model from December 2025 (C 2.33 MJ/K, U 51.7 W/K, tau about 12.5 h, cooling 0.65-0.70
C/h with heating off). The guest house has none.

## 4. Salon - requirements

S1. The comfort measure is the **mean of all valid temperature points** in the salon: both thermostats' room
    temperatures, the floor sensor and the ceiling sensor. Each point uses its latest value; a silent, flat or
    disagreeing sensor is left out by the existing sensor guard (`core/theria/sensor_guard.py`), which gets the
    salon sensors added. If fewer than two points are valid the zone is treated as "unknown" and not cut.
S2. An **opening schedule**, editable in Theria: weekdays 09:00-19:00, weekends 10:00-16:00 (local time).
    During opening hours the mean must be about **20 C** (setting `salon_open_target_c`, default 20.0, by Johan 2026-11-09).
S3. Outside opening hours the plan may let the salon cool (setting `salon_closed_floor_c`, default 17.0 -
    the builder's/our call, Johan gave no number; the water shut-off and pipes need nothing near frost
    levels).
S4. **Pre-heat:** before a dear period the salon may be heated above target, up to `salon_preheat_c`
    (default 21.0), so that it coasts through the dear hours and is back at the target only at the end of them.
    Heating up must also be planned so that the salon reaches the target by the time it opens.
S5. Johan's stated preference is to **err on saving too little**: the plan should cut the dear hours
    decisively but use the thermal model's uncertainty on the safe side, not the saving side. In practice: a
    margin of `salon_margin_c` (default 0.3 C) above the target in the prediction; and no cut if the zone's
    prediction error has recently been large (the same idea as `skill.py`).
S6. Both thermostats get the same setpoint (they are one room, two radiators).

## 5. Guest house - requirements

G1. Each of the five rooms has its **own comfort temperature** `T_room`, set in Theria (starting values: the
    current Tado setpoints).
G2. Around `T_room` the setpoint may move within a band `+/- band_c` (default 1.5 C, editable globally and per
    room), according to the price: upper side in cheap hours, lower side in dear hours.
G3. No opening hours and no schedule. A room outside the band (by its own temperature sensor) is never pushed
    further out: the lower bound wins over the price.
G4. A room set to `off` or manually changed by hand is left alone (section 7).

## 6. Control and its link to the pump plan

- The unit of work is the **hourly setpoint per radiator thermostat**, derived from the day's 15-minute Nord
  Pool prices (the same `price_feed.py` as the pump plan). Rule-based first (price rank within the day,
  salon with the cooling model for the coast); the builder may choose the exact form, but it must keep the
  guarantees in S1-S6 and G1-G4.
- The pump plan keeps its optimizer. The radiator block is changed from "no cut while any radiator calls" to
  "no cut while a radiator that Theria does **not** control calls, or while a controlled radiator's room is
  below its lower bound". The controlled radiators' calls then no longer block.
- **Why this saves energy:** the pump has no room sensor and makes heat for whatever the open loops and the
  radiators draw. Lowering the radiator setpoints removes their draw in dear hours; the pump plan can then
  also cut the flow. Savings are estimated (no per-zone heat meter): the model's heat by hour times price,
  against a flat-setpoint baseline, reported per zone.

## 7. Safety and manual changes

- A setpoint written by Theria is read back (as `LKExecutor` does with `read_until`); a refusal or a lag is a
  result, never an exception, and is logged with its reason.
- A change by hand (a value that is neither the last written nor pending, by at least 0.25 C) pauses
  control of that thermostat for one hour, as in `LKExecutor`.
- Leaving active mode, the kill switch, or Theria stopping restores the **base setpoints** (the schedule
  state in the thermostats before Theria took over; see open point 1).
- Unreadable entities (`unavailable`) are not controlled and are reported (the Netatmo/Tado entities were
  `unavailable` on 2026-10-14).

## 8. Shadow mode first (decided by Johan)

The first release computes the radiator plan and logs what it **would** write (`regulator_active: false` style
setting, e.g. `radiator_control: off | shadow | active`, only `off` and `shadow` built first). It writes nothing
to Netatmo or Tado. Logged per cycle: each room's measured temperature, the would-be setpoint and reason, the
salon mean and which points were used, the estimated saving. The UI shows this, and `active` is a later,
separate decision after Johan has read the log for some days.

Before `active` is built, a short manual **write test** is needed to settle open points 1 and 2 (it writes one
Netatmo and one Tado setpoint and reads how fast it arrives and whether it holds). It needs Johan's approval
at that time; it is not part of the shadow release.

## 9. UI and settings

- A page or section in the existing UI (with the plan settings, `settings_manager.py` overrides in
  `/data/plan_overrides.json`, applied live and resettable): the salon opening schedule per weekday group,
  the salon numbers (target, closed floor, pre-heat, margin), the guest house room list with `T_room` and
  band, and a switch `radiator_control`.
- A view of the radiator plan with the same conventions as the other plans: the day's prices, the would-be
  setpoint per hour, the measured temperatures, and the saving.
- Settings also go in `config.yaml` (schema, with defaults), as the other plan settings do.

## 10. Constraints

- Tado's cloud API has a daily call limit; five rooms hourly with reads and read-backs must fit with margin
  (the builder checks the limit that applies to Johan's account and counts calls). Netatmo has its own cloud
  rate limit and lag.
- Home Assistant is the only interface (`ha_client`); no direct cloud access.
- Follow `CODING_STANDARDS.md`: no commit without approval, no string matching on exceptions, no
  `hasattr`/fallbacks, deterministic.
- 15-minute price resolution; local time is Europe/Stockholm (`PlanSettings.timezone`).

## 11. Open points (to settle by measurement, not by Johan)

1. Netatmo in `auto`: does a written setpoint hold, revert to the schedule at the next schedule step, or
   time out? Which mode/service gives a persistent manual setpoint, and how are the schedule and base state
   restored? (`select.salong` = "Mitt schema" is the schedule.)
2. Latency and rounding of both clouds; whether Tado exposes the same half-degree step in practice; the call
   budget (section 10).
3. Which entity is the smoke-detector ceiling sensor (`butik_temperature_2` is the likely one).
4. The guest house's draw on the pump; the learned thermal rates per room (from InfluxDB history) to tune
   `band_c` and to know when a room is the limiting one.

## 12. Left to the builder

The exact rule that turns price into the hourly setpoint (rank, threshold or small optimization) within the
guarantees above; module layout (a new `radiator_executor.py` mirroring `lk_executor.py` is the natural
shape); storage of the schedule; how the radiator plan and the pump plan exchange the "blocked" state.

## 13. Calls made after the builder check (the writer's, not Johan's - change freely)

1. **Shadow and the pump block.** In shadow mode the pump plan is unchanged: the radiators still block cuts as
   today, since the real thermostats keep calling. The radiator plan only logs "would have lifted the block"
   per hour. The block is lifted only in `active`, which is a later decision.
2. **Saving estimate.** Baseline = the setpoints the thermostats have today (salon 21 C, guest house the Tado
   setpoints), held flat. The salon is estimated with its December 2025 model (heat by hour times price). The
   guest house has no model in v1: report degree-hours below and above `T_room` and an indicative SEK figure
   from a stated, crude loss coefficient, labelled as indicative. The estimate counts the radiators' own draw
   only, not the pump-plan cuts they would enable; that effect is shown separately as hours of lifted block.
3. **Salon control law.** The written value is the thermostat setpoint (the lever), chosen so that the
   predicted mean of all points reaches the target. The floor/ceiling sensors usually read below the
   thermostats, so the setpoint may exceed the target. Bound: `salon_max_setpoint_c` (default 22.0). Raising
   to reach 19 C in opening hours is allowed (it is what pre-heating before opening needs).
4. **Base setpoint.** Theria records each thermostat's setpoint when it first takes over and keeps a
   configured base per thermostat (`radiator_base_c`) used for restore. Whether Theria overrides the
   thermostats' own schedule or works with it follows from open point 1; the data model stores a full hourly
   setpoint per thermostat either way.
5. **Availability.** The shadow log records, per entity, whether it was readable each cycle, and the UI shows
   the share of cycles readable.
6. **UI.** A frontend exists (`frontend/`, React with Vite; `CLAUDE.md`'s "no frontend" is out of date). The
   radiator settings and plan go there, next to the plan settings.
7. **Manual-change detection in shadow.** Skipped (nothing is written). The log records the difference
   between each thermostat's setpoint and the would-be setpoint instead.
