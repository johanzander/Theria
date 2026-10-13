# IVT AirX / Rego 2000 - Knowledge Base

Heating-side behaviour of the heat pump at Johan's house, and the levers Theria can use.
Hot water (DHW) is out of scope for now; see [TODO.md](TODO.md).

Each statement is tagged:

- **[manual]** stated in a manual in `documentation/private/IVT/` (source in brackets)
- **[observed]** seen in this house's Home Assistant data on 2026-10-10/11
- **[unverified]** inferred; test before relying on it

## System

- IVT AirX air/water heat pump with HMC300 / Rego 2000 controller, interfaced through a Husdata H60.
- Control mode here: **outdoor-temperature controlled, no room sensor.** `sensor.indoor` reads 0.00 and
  `climate.room_sensor_influence` is 0. The pump only uses the outdoor sensor (`sensor.outdoor`).
- Heat is distributed to the first floor (floor heating, LK ARC hub with Arc Sense thermostats), boutique
  and guest house (radiators).

## How the pump picks the flow temperature

- **[manual]** The controller sets the flow temperature from a heat curve (outdoor temperature in, flow
  temperature out). Two curve styles exist: optimized and simplified (with base point).
  [HMC300 OM section 2.2; AirX 400 user manual 3.1]
- **[manual]** Without a room unit, the room setpoint is not a measurement. Setting it *changes the heat
  curve*, which changes the hot-water temperature and thereby radiator / floor temperature.
  [HMC300 OM section 8, first FAQ]
- **[manual]** With a room unit, the measured vs wanted room temperature would adjust the curve. We have none.
- **[manual]** Optimized operation (no time program) heats continuously to the set temperature.
  [HMC300 OM glossary]
- **[manual]** Maximum flow temperature with heat pump only is 62 C. [Install AirX table 3]
- **[observed]** `sensor.heating_setpoint` (reg 0107) is the pump's *computed* flow target, read-only in
  practice. It was 32 C at 12 C outdoors and drifted to 33 C overnight as outdoor fell to ~10.5 C.
  Not set by any Home Assistant automation.
- **[unverified]** `sensor.heating_setpoint_2` (39 C) is probably the second heat circuit or the
  uncompensated target; unknown.

## Levers writable through the H60

| HA entity | H60 register | Meaning | Value 2026-10-11 |
|---|---|---|---|
| `climate.room_temp_setpoint` | n/a | Wanted room temperature; shifts the curve | 18.0 C (range 13-29) |
| `climate.room_sensor_influence` | n/a | Weight of room sensor on the curve | 0 (range 0-10) |
| `climate.heat_set_1_curvel` | 0205 | Curve **base point** temperature, circuit 1 | 21.0 |
| `climate.heat_set_2_curver` | 0206 | Curve **end point** temperature, circuit 1, radiator system | 15.0 |
| (not in HA) | 020E | Curve end point, circuit 1, floor heating (DesignR) | unknown |
| (not in HA) | 02F1 | Room controller emulation (indoor temperature fed to pump) | not used |

[register meanings: Husdata C10 register map]

- **[unverified]** Plan: change `room_temp_setpoint` by 1 C and measure the change in `heating_setpoint`
  (degrees of flow temperature per degree of room setpoint).
- **[per Johan]** CurveL / CurveR are how the H60 sets the curve, but no effect on the IVT/Rego has been
  seen, so they are **not used**. The house heat curve is changed **manually on the pump**. The
  values shown in HA (21 / 15) are therefore not the active curve. History of manual curve changes
  is not recorded anywhere; it has to be inferred from data (see below).
- **[manual]** Husdata can emulate an RC100/CR10/TR111 room controller on Rego 2000/3100 for circuit 1
  (register 02F1), so we could feed a *real* indoor temperature and let the pump's own room influence
  work. No other room controller may run in parallel; updates must arrive at least every minute or the
  pump alarms (informational). [Husdata H66 documentation; C10 register 02F1; only verified for H66, H60
  unconfirmed]
- Existing but disabled: `automation.set_room_temp_setpoint` (never triggered) maps
  `sensor.set_temp_value` to `room_temp_setpoint` and to Tado/Netatmo setpoints with fixed offsets.
  Helpers `input_number.*_temp_offset_*` (all 0) and `temp_set_base_value` (19) belong to the same legacy
  price control.

## Heat curve - empirical (InfluxDB, 2022-10 to 2026-10)

Analysis of hourly means of `sensor.outdoor`, `sensor.heating_setpoint`, `sensor.radiator_forward`
(home-analytics InfluxDB, measurement = entity_id, field `value`). Hours with setpoint < 15 C
(summer / off) excluded. The curve is changed manually on the pump, so changes are inferred, not logged.

- **[observed]** `heating_setpoint` is a near-linear function of outdoor temperature. Winter 2025/26
  (Dec-Mar), 0-12 C outdoors: `setpoint ~ 42.0 - 0.68 * T_out` (r = -0.82, residual std ~1.9 C).
  Extrapolation below 0 C is thin (few hours below -2 C) and unverified.
- **[observed]** The pump uses a lightly smoothed outdoor temperature: best fit with a 3-6 h average
  (r = -0.82) vs 1 h (-0.79), 24 h (-0.78), 72 h (-0.71). It reacts within hours, not days.
- **[observed]** Setpoint is the heating target irrespective of the hot water valve; values are the same
  in heating and hot water periods (valve data is sparse).
- **[observed]** The curve level has been changed several times (steps in weekly residual vs the
  2025/26 reference curve): about +6 C in winter 2023/24, +3 C in Oct-Dec 2024, about 0 / -1.5 C from
  Jan 2025, +1..+3 C in Nov-Dec 2025, and a steady **-1.2 to -1.5 C since Feb 2026**. The current curve
  is therefore about `40.8 - 0.68 * T_out`. At 12 C outdoors this gives 32.6 C, matching today's 32 C.
  Caveat: residuals assume a straight line; if the real curve bends, seasonal bias remains.
- **[observed]** Median setpoint by outdoor temperature, seasons 2024 and 2025 (the 2022/23 seasons were
  higher): 0 C -> 42; 4 C -> 39; 8 C -> 36-37; 12 C -> 33-35.
- **[observed]** Actual `radiator_forward` follows the setpoint in heating hours (Nov 2025 - Apr 2026,
  median error -0.2 C, p10..p90 -5.7..+3.7 C). Compressor speed averages 7 % at 12 C and 11-13 % at
  -3..0 C outdoors, so the pump runs at low speed for most of the season.
- **Implication [unverified]:** Theria cannot change the curve shape without the pump menu, but it can shift
  the whole curve through `room_temp_setpoint`. Johan will confirm the shift per degree from this
  history rather than by a live test; no `room_temp_setpoint` changes are available in Influx yet to
  isolate it (check whether the old automation ever wrote it).

## Room setpoint as a lever - measured from history

`climate.room_temp_setpoint` (attribute `temperature`) was driven by an HA price automation
(`sensor.set_temp_value`, base + offsets by price level) from 2022-10 to about 2025-12. It moved between
~14 and 24 C, often several times per day. Later it was left at 18-19 C (since Jan 2026 it has barely changed).

- **[observed]** 305 clean steps (room setpoint stable 6 h before and 2-7 h after, outdoor change
  corrected with the -0.68 C/C curve slope). Each degree of room setpoint moved `heating_setpoint` by a median
  of **1.9 C** (IQR 1.7-2.1, n = 227 steps of 1.5 C or more). The ratio is the same in 2022, 2023, 2024 and
  2025 (1.89-1.95), so the effect is stable and independent of the manual curve level.
- **[observed]** The response shows within an hour (first-difference regression gives 1.8 at lags 1-6 h).
- **[observed]** The actual flow temperature follows `heating_setpoint` (see above), so +1 C on the room
  setpoint is roughly +1.9 C on the flow temperature. Compare: one degree of outdoor temperature moves it
  only 0.68 C, so a +1 C room setpoint equals about -2.8 C outdoors.
- Meaning for Theria: the room setpoint is a strong, fast, predictable lever on flow temperature. How fast
  the floor and rooms respond (thermal lag of the concrete) is not yet measured.

## Data quality notes

- H60 sensors are noisy; Johan filters spikes in `~/GitHub/home-analytics` (`backend/api.py`:
  `_remove_outliers`, gap-spike repair, `_enforce_monotonicity`). That pipeline is aimed at **cumulative
  energy counters** (`compr_consump_tot`, `supp_energy_tot` etc. in `backend/sensors.yaml`). Reuse it for any
  energy or COP numbers in Theria.
- Temperature series used here are mostly clean, but `sensor.outdoor` in InfluxDB has isolated garbage
  (max 240 C). Analyses above drop values outside -35..45 C and use a 3 h rolling mean.
- InfluxDB (home-analytics config): InfluxQL, measurement = entity_id, field `value`; climate entities store
  attributes as fields (`temperature`, `current_temperature`). `heating_setpoint` is stored on change only, so
  forward-fill before resampling.

## First floor - measured behaviour (InfluxDB LK data from 2025-12-17, Open-Meteo archive for sun)

Rooms = Bibliotek and Vardagsrum LK thermostat temperatures; LK setpoint is 19.5 C almost all the time.
Sun = Open-Meteo `shortwave_radiation` for lat 55.578 / lon 12.927 (modelled, not measured on site).

- **[per Johan] Confounder: tile stove (kakelugn).** Johan burned a lot in the stove last winter, which
  warms the first floor on evenings without any other trace in the data. This affects the winter
  undershoot numbers (the deficit would have been larger without it) and the shoulder-season sun analysis
  (evening room peaks at 16-18 UTC may partly be stove, not sun). The sun table and lag above must be
  re-checked with stove evenings excluded (no stove sensor exists; infer from evening rises
  not explained by irradiance).
- **[observed] Winter undershoot.** Dec 2025 - Feb 2026 the rooms were *below* the LK setpoint: weekly means
  Bibliotek 17.3-19.3 C, Vardagsrum 16.2-18.3 C (monthly mean deficit Jan -0.8 / -2.4, Feb -1.3 / -2.7 C).
  The flow temperature then was about 39-40 C at 0-1 C outdoors with the room setpoint at 18-20 C. So in
  winter the current curve plus the room setpoint did not reach 19.5 C in the first floor, mainly
  Vardagsrum. Needs a decision before the cold sets in (see TODO).
- **[observed] Sun overheating in shoulder seasons.** In Mar-May and Sep-Oct the rooms sit at the
  setpoint at night and in the morning (excess about 0), then exceed it in the afternoon: mean excess peaks at
  +1.5 C (Bibliotek, 16-17 UTC) and +1.4 C (Vardagsrum, 18 UTC).
- **[observed] Size of the effect (days with 3-15 C outdoors, n = 113 days).** Rise from the 05-07 UTC
  level to the afternoon peak:

  | Daily irradiance (kWh/m2) | Bibliotek rise | Vardagsrum rise |
  |---|---|---|
  | < 1 | 0.6 C | 0.7 C |
  | 1-2 | 0.9 C | 0.9 C |
  | 2-3 | 2.4 C | 2.0 C |
  | 3-4 | 2.7 C | 2.4 C |
  | 4-6 | 2.3 C | 3.6 C |

  Linear fit: rise = 2.10 + 0.36 * kWh/m2 - 0.16 * T_out (Bibliotek, R2 0.23) and
  1.65 + 0.56 * kWh/m2 - 0.16 * T_out (Vardagsrum, R2 0.43). The step above ~2 kWh/m2 is clearer than the
  linear fit. Vardagsrum responds more to sun than Bibliotek.
- **[observed] Lag.** The rooms follow irradiance with a delay: correlation peaks at **4 h** (0.72 Bibliotek,
  0.81 Vardagsrum; values 0.63-0.71 at 2-3 h, 0.66-0.78 at 5 h). Sun at 13 UTC shows up as peak room temperature
  at about 16-18 UTC.
- **[unverified] Flow-temperature lag not measurable.** The room setpoint changed only in the autumn of 2025,
  before LK logging, and in winter room temperature and flow temperature correlate at most 0.2 (LK valves
  confound it). A new controlled test is needed to measure the concrete's response to the flow temperature.
- **Implication [unverified]:** On a sunny day the lever is *not* the pump flow temperature. The LK valves
  are mostly closed in the afternoon, so flow temperature hardly matters then. The gain is to avoid heating
  in the hours before sun arrives (lower the LK setpoint or the room setpoint ahead of time, using the
  Open-Meteo forecast with a ~4 h lead) and to let the sun do the work.

## Room setpoint vs pump electricity (first estimate)

- **[observed]** The compressor practically never stops in heating mode, even with the room setpoint at 13-15 C:
  <1 % of heating hours had speed < 1 % (3 years, 0-10 C outdoors). It modulates instead. Mean compressor
  speed was ~7.2 % at room setpoint 15 vs ~9.2 % at 22 (uncontrolled for outdoor temperature).
- **[observed, rough]** Hourly increments of `sensor.compr_cons_heating` (spikes filtered with a 5-point
  median) regress as `kWh/h = 0.31 - 0.15 * T_out + 0.13 * RTS` (n = 5822 heating hours, 0-10 C outdoors),
  i.e. about **7 % more compressor energy per degree of room setpoint** around 1.8 per hour. Binned: ~1.2-1.6 at
  RTS 14-16, ~2.0 at 19-22. Unit of the counter is uncertain (values ~72 000 in old data vs ~700 now) and the
  fit ignores solar and wind; treat as order-of-magnitude.
- **Implication [unverified]:** The pump cannot be "turned off" with the room setpoint; the saving is modulation
  (roughly 7 % per degree, so ~35 % lower heating energy at room setpoint 13 vs 18). Price-driven shifting
  therefore means *less* energy in expensive hours and *more* (pre-heat) in cheap ones, not on/off.
- `sensor.compressor_speed_2` (kW) is not usable in InfluxDB (zero / missing).

## Floor construction, LK design and the flow temperature limit

Sources: `documentation/private/Reviderad beräkning 3 LK023393.pdf` (LK calculation, revised 2026-01-12),
`documentation/värmebehovsbedömning_ Vattnet 1, Malmö Mailid_0074000001094.eml` (LK sales engineer, 2022-06-01),
`documentation/private/VÄRMEBEHOVSANALYS - VATTNET 1.pdf/.xlsx` (heat demand analysis, 2026-01-01).

- **[manual]** Ground floor: 16 mm solid oak parquet glued on 10-12 mm ROT board over LK groove boards in a timber
  joist floor (system G16, floor resistance Mtot 0.200 m2K/W); tiled rooms (halls, bath) on LK 8 mm strips
  (Mtot 0.020). [LK calculation]
- **[manual]** LK did not calculate heat demand: it designs for 50 W/m2 unless told otherwise, with a design indoor
  temperature of 20 C (22 C bathroom). It recommends at most 25 mm wooden floor because "the system gets slow"
  (trogt) and says a shunt is only needed if other heating shares the same system. [LK mail]
- **[manual] Design flow temperature 45 C** (mean return 39.3 C), 8 circuits, 76 m2, 4204 W delivered against
  4788 W needed. Parquet rooms: floor surface 25.1-25.2 C; tiled rooms 31-31.5 C. Living room (V-rum) circuits
  are short by 175, 223 and 175 W = **573 W** at design conditions. [LK calculation]
- **[manual]** Heat demand at DUT -9.6 C and 20 C indoors: 7889 W, 63.2 W/m2 for the whole ground floor; kitchen 74.5,
  dining/living 70.6, hall #1 53.6, hall #2 92.4, guest room 53.5, library 53.9 W/m2. [Heat demand analysis]
  The kitchen and hall #1 circuits (1:4, 1:6) are not in the revised LK calculation.
- **[per Johan] 45 C is the maximum flow temperature.** Nothing above LK's design value is acceptable for the
  floor (earlier drafts used 50 C, derived from an assumed 27 C surface limit; that number appears in no
  document and was removed). The planner default is `max_flow_c = 45`.
- **[derived]** At 45 C flow the parquet surface is 25.1 C at 20 C room temperature and about 25.9 C at 21 C, so
  neither a 26 C nor a 27 C surface limit is reached (general guidance for solid wood is 27 C: Bauhaus guide
  https://www.bauhaus.no/media/pdf/4005461A.pdf; the manufacturer's warranty is binding and is not in the
  project documents). Daily surface swing should stay under 5 C: 13 C of flow swing is about 3 C at the surface.

### Where the floor reaches its limit (derived from the two documents above)

Parquet floor capacity at 45 C flow and 20 C indoors: **54.6 W/m2** (living room: 2531 W demand - 573 W LK
shortfall over 35.85 m2). Demand scales with (20 C - outdoor) / 29.6 K. Linear model, 20 C indoors:

| Room | Demand at -9.6 C | Outdoor temperature below which 45 C flow is not enough | Shortfall at -9.6 C |
|---|---|---|---|
| Guest room | 53.5 W/m2 | not reached | 0 |
| Library | 53.9 W/m2 | not reached | 0 |
| Dining/living room | 70.6 W/m2 | **-2.9 C** | 573 W |
| Kitchen (parquet assumed, circuits not in LK calc) | 74.5 W/m2 | **-1.7 C** | about 570 W |
| Hall #1 | 53.6 W/m2 | not reached | 0 |
| Small hall (tile) | 92.4 W/m2 | not reached | 0 |

- **[observed]** Hours per heating season below those outdoor temperatures (`sensor.outdoor`, hourly means):
  below -1.6 C: 66, 201, 51, 259 hours (seasons starting 2022-2025); below -2.8 C: 35, 137, 28, 132 hours; below
  -5 C: 4, 72, 7, 16 hours; nothing below -9.6 C (coldest hour -8.8 C, 2024-01-07). That is 0.4-3 % of the heating
  season for the living room, and at most about 570 W short at the design extreme: a tile stove covers it.
- **[derived]** Flow temperature the living room needs: `42.4 - 0.98 * T_out` C (20 C indoors). The pump curve at room
  setpoint 18 C gives `40.8 - 0.68 * T_out`. They agree at +5 C, the pump is 1.6 C short at 0 C, 2.5 C at -3 C and
  4.5 C at -9.6 C. This matches the winter undershoot in the living room: a higher room setpoint (about +1) or a
  steeper curve is needed, and above -2.9 C the 45 C cap is not the limit. Below it, the cap is.
- **[per Johan] Planned change: kitchen and large hall.** Today the kitchen has only one small radiator and is
  under-dimensioned (the kitchen and hall #1 circuits 1:4/1:6 are not in the revised LK calculation). The plan is tile
  in the whole kitchen and the large hall with water-borne underfloor heating, and Johan hopes the kitchen's heat will
  spread to the living room through the fairly open floor plan. Not yet built, so none of the numbers above include
  it. Tile has a much lower floor resistance than parquet (Mtot 0.020 vs 0.200 m2K/W, LK calculation), so the
  kitchen should reach its demand with a lower flow temperature and may stop being the limiting room; the open
  plan could also lower the living-room deficit. Re-calculate after the work.
- **Implication for the planner:** the pump's flow target at room setpoint 18 C reaches 45 C at about -6 C outdoors,
  so in normal Malmo weather the cap leaves room for pre-heating; in cold snaps it is what limits the plan.

## Fitted models (calibration, 2026-10-11)

Fitted by `core/theria/plan/calibration.py` on 1095 days of history from the local InfluxDB (hot-water time excluded
by the share of each hour the valve was on, H60 spikes removed). `scripts/calibration_report.py` reruns it; the add-on
refits every 24 hours and keeps the previous models if a fit is rejected. All values **[observed]**.

- **Heat curve (good):** `flow target = 40.67 - 0.709 * T_out` at room setpoint 18.0 C (+1.9 C per C of room setpoint),
  fitted on 1287 heating hours below 12 C in the last 240 days, RMSE 0.52 C. Agrees with the earlier winter estimate
  (`40.8 - 0.68`).
- **Heating electricity (usable):** `-0.73 - 0.206 * T_out + 0.223 * RTS` kWh/h on 8453 pure-heating hours (RMSE 0.74
  kWh; the counter has 1 kWh resolution). The room setpoint effect is about 0.22 kWh/h per C (roughly 11 % of the
  mean hourly consumption per degree), higher than the first rough 0.13 because hours with a short hot-water block no
  longer dilute it.
- **Room model, replaced (2026-10-11).** The hourly two-state room model failed the honest test (free-running 24 h ahead
  it was 1.99 C off against 0.71 C for "no change"; the prior was no better), so it was dropped. The cause: hour by hour the room is
  moved mostly by things the data does not show (sun, stove, people, doors), the house is very slow, and in the LK-era
  data the flow temperature simply follows the outdoor temperature, so the floor's effect cannot be told from the heat
  loss. See the next bullet for what replaced it.
- **Room response to the flow temperature (daily model, `core/theria/plan/daily_room.py`).** `R' = k_out (T - R) +
  k_flow (F - R) + k_sun * sun + b` per day, fitted season by season on daily means of the room, outdoor temperature,
  flow target and sun, free-run 1-3 days ahead, using only heating days (daily mean outdoor temperature at most 12 C;
  summer and shoulder days, where flow means nothing, biased the first version). The room series is the LK sensors where
  they exist and the long-history Sonoff sensor (vardagsrum) before that, shifted by the median difference to the LK mean
  (about 1.0 C; 1.7 C against LK Vardagsrum alone) and scaled to the LK sensors' day-to-day swing (the Sonoff swings
  about 25 % more). The price-controlled years 2022-2025 (flow temperature varied a lot independently of the weather)
  identify the response. The splice with LK is in the middle of the 2025/26 season.
  Per heating season (1 July to 30 June), gain in C of room per C of flow in the long run / rate per day / heating-day windows:
  2022/23: 0.39 / 0.16 / 75; 2023/24: 0.45 / 0.48 / 56; 2024/25: 0.47 / 0.22 / 51; 2025/26: 0.45 / 0.14 / 150 (2026/27 has too
  few cold days yet). The seasons agree on the gain (0.39-0.47) and, within a factor of 5, on gain x rate (the quantity
  that sets the first days). **Median: one degree of flow moves the room about 0.45 C in the long run (so about 0.85 C per
  degree of room setpoint), and the room closes about 19 % of the gap per day (time constant about 5 days).** The two Sonoff-only
  seasons may overstate the gain slightly (a different sensor, scaled to the LK mean, not measured on it).
- **What this model is NOT:** it does not predict the room temperature. Leave-one-season-out, 1/2/3 days ahead it is off
  by 0.48 / 0.67 / 0.83 C against 0.47 / 0.65 / 0.77 C for "no change". It is accepted because the response to the flow
  temperature is present in every season and the seasons agree (largest/smallest gain at most 4x), not because it
  predicts well. The plan therefore starts from the measured room temperature, keeps the baseline (a flat setpoint) where
  the room is now (`anchored_room_params`), and uses the model only for how much the plan's deviation from that
  baseline moves the room. The short-term wiggle of the room (stove, sun, people) is not modelled.
- **Consequence for the optimisation:** a few hours of lower or higher flow barely move the room (a 3 C room setpoint
  change for 4 hours is about 6 C of flow and moves the room about a hundredth of a degree), so a short price peak can
  be avoided at almost no comfort cost; what matters is the daily average heat input. Lowering the setpoint to the
  minimum for a whole day sags the room about 0.4 C in the model, and the sag has to be paid back (the house heats up as
  slowly as it cools), so the plan keeps the day energy neutral.
- **The plan's objective after Johan's review (2026-10-11):** the plan covers the known prices only (about 12-36 h), must end
  the horizon with the room where it started (like the BESS planner's state of charge), pays 0.1 SEK per degree-hour for
  every degree the setpoint is away from the base setpoint (a stand-in for the salon and guest house, which react faster
  and are not modelled), and so varies the setpoint over the day instead of cutting the heat for many hours. Live example
  (2026-10-11, room 20.1 C): setpoint dips to 15 for 2 hours in each price peak, rises to 22 for 4 hours in the cheap
  night, stays at 18 otherwise; the room stays within 0.1 C; mean setpoint 17.8; modelled saving about 15 % of the
  heating cost for the 29 h (an estimate: it is not measured).
- **What decides if the plan can be trusted:** `GET /api/model/health` shows the response fit (`models.room.fit.response`
  incl. per-season gains and the cross-validation) and, in shadow mode, `scores` (plan error 1, 6 and 24 h ahead against
  readings). The energy saving the plan reports is a model estimate (energy per setpoint degree from history) and has not been
  measured.

## Comfort sensors and their guardrails (2026-10-11)

Code: `core/theria/sensor_guard.py`, `GET /api/comfort-sensors`, banner in `backend/static/js/health-banner.js`,
options in `config.yaml` (`sensor_guard`). All values **[observed]** unless noted.

- **Sensors:** LK thermostat sensors (the reference LK regulates on, cloud-polled) and Sonoff Zigbee sensors as a second
  opinion: Vardagsrum Sonoff `sensor.sonoff_a480045bf5_*` (follows LK Vardagsrum, correlation 0.90, about +1.7 C
  warmer) and Bibliotek Sonoff `sensor.sonoff_a480050818_*` (follows LK Bibliotek, correlation 0.92, about +1.7 C).
  The other two Sonoffs (`a4800560df`, `a4800560e1`) are not watched; their rooms are unknown.
- **Both Sonoffs have been dead for months:** last numeric readings 13 Feb 2026 (vardagsrum, battery 100 % at the end) and
  15 Feb 2026 (bibliotek, 93 %); the third (`a4800560e1`) last reported 30 Aug 2026 at 9 % battery. Johan believes the
  batteries are flat. The battery percentage is unreliable (it showed 100 % until the sensor went quiet), so the guard
  trusts silence more than the percentage. Historically the batteries lasted about a year and were replaced in
  Nov 2023, Nov 2024 and Nov 2025.
- **Reporting rhythm:** the vardagsrum Sonoff reported about once an hour (median gap 60 min, longest 449 min). The LK
  sensors change value roughly every 30-45 min (median), 99 % within 4-6 h, longest about 11 h; their HA entities refresh
  every 5 min, so `last_changed` (value changes) is the signal, not `last_updated`.
- **Rules:** unavailable / non-numeric = fault; implausible (< -10 or > 45 C) = fault; no value change for 8 h (LK) or 6 h
  (Sonoff) = warning, 16 h / 12 h = fault; battery <= 25 % = warning, <= 10 % = fault; signal <= -85 dBm = warning
  (shown, not notified); a sensor that drifts more than 1.0 C from its usual difference to the room's reference over
  6 h = warning. Home Assistant persistent notifications are sent when a problem is seen in two scheduled checks in a row
  (10 min apart) and dismissed when it clears; silent *warnings* and weak signal are shown but not notified.
- **Room temperature for the plan:** each usable sensor is corrected to its room's reference (median difference over 7 days)
  and the rooms are averaged with equal weight. A low battery alone does not discard a valid reading. No usable sensor means
  no new plan (the previous one stays). Right now only the two LK sensors are usable.
- **Caveat:** after a Home Assistant restart every entity's `last_changed` is reset, so a stuck sensor can stay hidden for
  up to 8-16 h after a restart.

## Electric heater and compressor: what the 2022-2026 history says (2026-10-11)

Hourly analysis of the heating hours below 12 C outdoors (about 1160 hours with energy counters, valve in heating
position), from the years when the room setpoint was moved by hand and by the price automation (typically +/- 2 C around its
own 3-day median, sometimes more). All values **[observed]**; the sample for deep or long drops is small.

- **The compressor never stopped in heating mode:** speed below 1 % in 0 of 1162 hours, also in the hours with the
  setpoint 2-3 C below its median (31 h) and in the 12 hours after setpoint drops of 2-3 C (537 h), 3-4 C (215 h) and over 4 C
  (37 h). The fear that a deep cut makes the pump shut down is not supported by this data (but drops longer than a few hours
  and deeper than -3 C are rare in it).
- **The electric heater (aux) is a cold-weather, high-flow-target effect, not a rebound effect:**

  | Flow target (heating_setpoint) | Hours with aux > 0 | Mean aux |
  |---|---|---|
  | below 40 C | 0 % | 0 kWh/h |
  | 40-44 C | 1.6 % | 0.016 kWh/h |
  | above 44 C | 21 % | 0.25 kWh/h |

  By room setpoint at -10..3 C outdoors: 13-18.5 C: 1.5 % of hours; 18.5-20 C: 9 %; 20-22 C: 33 % (0.36 kWh/h); 22-26 C:
  65 % (0.94 kWh/h). At 3-12 C outdoors the heater is almost never on, whatever the setpoint (0-2 %). After a setpoint rise of
  2 C or more aux use is 2.5 times the quiet-period level, but that is largely the high setpoint that follows the rise.
  The heater cost is not in the compressor energy counter that the plan's energy model is fitted to.
- **Consequence for the plan:** a boost of the flow temperature into the 44+ C range in cold weather is what triggers the
  heater, and it removes the saving (aux has COP 1). The plan therefore (a) keeps the setpoint within +/- 2 C of the base
  (16-20 C, like the manual control), (b) may raise it by at most 1 C per hour (lowering is not limited), and (c) does not ask
  for a flow target above `boost_flow_limit_c` = 42 C unless the baseline flow already is higher (cold weather): then it
  may only cut. The floor limit of 45 C stays.

## Heating vs hot water

- **[manual]** By default hot water has higher priority and interrupts a heating demand. With
  "VV-vaxeldrift" (DHW alternating mode) both have equal priority and take turns in the configured time
  windows. [HMC300 OM section 5.7]
- **[observed]** `binary_sensor.switch_valve_1` (reg 1A07): off = heating circuit, on = hot water. Over
  the last ~32 h it was on 3.7 h (11 %), in 23 short blocks of 10-45 min. Heating is therefore blocked
  ~11 % of the time at 11-15 C outdoors. Expect this to rise when it gets colder.
- **[unverified]** The hourly sawtooth in `radiator_forward` (about 26 C to 39 C) probably follows these hot
  water cycles (residual heat leaving through the heating circuit, see FAQ in HMC300 OM section 8). Not
  confirmed.

## Defrost and minimum energy in the heating system

- **[manual]** Above +5 C outdoors: fan at max, compressor speed limited, heat production continues (SSD).
  Below +5 C: hot-gas defrost, the refrigerant circuit reverses and the heating system cools slightly.
  [Install AirX 3.7]
- **[manual]** To avoid short cycles, incomplete defrost and nuisance alarms, enough energy must be
  stored in the system. Floor heating without buffer tank: the largest room must *not* have room
  thermostats, and at least 30 m2 of floor must be regulated by a room unit. Radiators without buffer:
  at least four 500 W radiators in an unshunted circuit. [Install AirX 3.3]
- **Implication for Theria [unverified]:** if LK closes all first-floor loops at the same time the
  system may lack the minimum water volume for defrost. We have not checked whether this house has a
  buffer tank.

## Other inputs

- **[manual]** Smart Grid boost (0-5 C) and PV boost inputs exist on the controller (forced boost
  2-5 C). [HMC300 OM sections 5.14, 5.15] **[observed]** `climate.external_control` and
  `external_control_2` exist in HA; not yet identified.
- **[per Johan]** `sensor.limit_compressor` and `sensor.limit_auxiliary_heat` are external inputs to limit
  or disable compressor and electric heater. They are not wired up.
- **[observed]** `sensor.add_heat_status` is 0 right now. `aux_cons_heating` (16 kWh) is small, so the
  heater is not driving the heating bill.

## First floor control chain (LK)

- Pump flow temperature follows outdoor temperature only. The LK system then decides per area (Bibliotek,
  Vardagsrum) how much of that flow reaches the floor, by opening or closing actuators.
- **[observed]** LK cloud: Bibliotek has 5 actuators, Vardagsrum 3. Hub structure endpoint returns
  `actuators: [{n, s}]` per thermostat (`s: 0` assumed closed). The Home Assistant integration does not
  expose this; `hvac_action` there is just `current < target`.
- **[observed]** On a sunny day (2026-10-10) both rooms rose from ~19.3 C to 21+ C with thermostats idle
  and only 12-15 C outdoors, so solar gain alone exceeded the 19.5 C setpoint by 2 C.

## Sources in `documentation/`

- `IVT/HMC300Rego2000_OM.pdf`: controller user manual (curve concept, priorities, FAQ)
- `IVT/Install_AirX_6720810160_01.pdf`: installation manual (defrost, minimum volume, limits)
- `IVT/Anvandarmanual_AirX400-400S_AirModule.pdf`, `IVT/Anvandarhandl_AirX_6720810156_01.pdf`
- `IVT/Install_AirBox_6720810154_01.pdf`, `IVT/Capacitetsdiagram Air X 5-17.pdf`
- `documentation/private/C10.pdf`: Husdata H60/H66 register map
- Not yet read in depth: AirBox installation manual, capacity diagrams, the heat demand analysis files.
