# Theria Algorithm Explained

How Theria decides what to do with the heat pump, what the house needs, and what the energy and money work out to. For how
the code is built, see [SOFTWARE_DESIGN.md](SOFTWARE_DESIGN.md).

Sources are marked **[manual]** (datasheets, design documents), **[derived]** (calculated from those), **[observed]**
(measured in this house) and **[estimate]** (assumptions are named). The temperatures in sections 2-5 are settings; they are
collected in [section 6](#6-the-temperatures-are-settings).

## 1. What kind of system this is

The house is the store. Heat bought in a cheap hour stays in the building for a while, so a dear hour can use less. Theria
has two levers: the pump's room setpoint, which moves the flow temperature for every circuit, and each LK thermostat, which decides
whether that room's floor loops are open.

The pump has no room sensor. A raised setpoint keeps costing energy as long as it stands, so the plan always returns to the
base setpoint after a window. Lowering the setpoint alone does not stop the consumption, because open floor loops keep
drawing heat. That is why a sell also closes the loops
([design](superpowers/specs/2026-11-09-closed-loop-shift-design.md), [part 1](superpowers/specs/2026-11-09-part1-floor-heating-design.md)).

## 2. Inputs

Every 15 minutes the plan gets:

- **Prices** per hour from local midnight to the end of what Nord Pool has published.
- **Outdoor temperature and sun** from Open-Meteo, shifted by the difference to the measured outdoor sensor now.
- **The measured room**, from the sensors that pass the sensor guard.
- **The salon's measured mean** and the window that has started, for the guards.

The plan has one value per hour. Replanning every 15 minutes corrects it from fresh measurements.

## 3. How the plan decides (`heat_shift`)

1. **Window length comes from the house.** The time the fastest-cooling zone takes to go from +swing to -swing, between 1 and
   8 hours, at the horizon's mean outdoor temperature:

   ```text
   cooling = loss_per_h * (reference room - outdoor), taken for ground floor and salon
   hours   = round(2 * shift_room_swing_c / fastest cooling)
   ```

   At a swing of 0.5 C this gives 2 h at -10 C, 3 h at -5 to 0 C, 4 h at 5 C, 5 h at 7.5 C and 8 h at 12 C.
2. **Windows come from the prices.** Slide a window of that length over the prices. Local minima are buys, the following
   maxima are sells. Pairs are dropped, smallest first, until every remaining pair is worth more than
   `shift_min_gain_sek_per_kwh`.
3. **Balance.** Unmatched hours are given up, the least valuable first, until at most 10 % of the moved hours are unmatched.
   Heat is moved, not cut.
4. **Setpoint per hour.** Base, base minus `shift_rts_c` in a sell, base plus `shift_rts_c` in a buy. A sell wins an overlap.
5. **A buy only where the heat is needed.** Moving heat to cheaper hours is not a goal in itself. A buy stands only if it ends just before a sell and some room would
   otherwise end the sell under its closing target ([design and review](superpowers/specs/2026-11-09-buy-only-when-heat-can-be-stored.md)). A sell left without a buy must
   beat the price of the hours after it. On most days this leaves a sell with nothing before it.
6. **Thermostat per hour.** A sell is `closing`: the thermostat stands at the room's lower limit plus `lk_close_offset_c`, so the
   loops close and reopen by themselves if the room falls that far. A buy that ends at most `shift_buy_max_gap_h` before a sell is
   `opening`: the thermostat stands at `lk_upper_c`, unless the room is within `lk_opening_margin_c` of it. A buy further from its sell
   does not stand (step 5). Otherwise `normal`, at `lk_base_c`.
7. **Consequences for each zone** are worked out against the flat setpoint:

   ```text
   delta(next) = delta + gain * flow_delta - (gain + loss) * delta
   ```

Example, 9 Nov 2026, mean 7.1 C outdoors, window 5 h: setpoint 19 at 23-03, 17 at 15-19 (5.56-7.44 kr/kWh), otherwise 18. The
curves give the ground floor at most +0.09 / -0.05 C and the salon +0.07 / -0.04 C against the flat setpoint. Modelled saving
4.8 kr that day, not measured.

## 4. Rules around the plan

- **A started window runs to its end.** A replan inside it keeps it.
- **Each LK room guards itself** through its thermostat.
- **Measured guards end a sell for the pump:** the salon under its limit (19 C while Netatmo asks for at least 20 C, otherwise 17 C), or the salon's temperature unknown. The thermostats stay
  closing to the end of the window, so that a room near its limit is not opened at the dearest price.
- **An ended window stays ended** until its planned end, so that a reading around a limit cannot switch it on and off.
- **A hand change on a thermostat** pauses that room and ends the window for it.
- A raise is never limited by a guard.
- The setpoint range is scaled by how well the model has predicted over the last three days (`skill.py`).

## 5. From plan to pump

The current hour's setpoint is written in active mode, with the limits in
[SOFTWARE_DESIGN.md](SOFTWARE_DESIGN.md#planexecutor-and-safetymonitor-planexecutorpy), and each thermostat gets its target for the hour.

The other strategy, `optimizer`, chooses the setpoint per hour by dynamic programming over room and slab, including
closing the loops at the lowest setpoint. It is kept but is not the default.

## 6. The temperatures are settings

| Setting | Default | Meaning |
|---|---|---|
| `rts_base_c` | 18 | setpoint outside windows |
| `shift_rts_c` | 1 (at most 2) | step in a window |
| `shift_room_swing_c` | 0.5 | allowed swing, sets window length |
| `shift_min_gain_sek_per_kwh` | 0.5 | smallest price difference |
| `rts_min_c`, `rts_max_c` | 15, 22 | range, scaled by authority |
| `max_rts_step_per_hour_c` | 2 | largest raise per hour |
| `target_c`, `floor_c` | 19.5, 18.5 | day target and limit, ground floor |
| `night_target_c`, `night_floor_c` | 18.5, 17.5 | the same at 00-06 |
| `extreme_floor_c` | 17 | limit in extreme-price hours |
| `ceiling_c`, `lk_limit_c` | 22 | room limit; thermostat under the optimizer |
| `lk_base_c` | 19.5 | normal thermostat target |
| `lk_lower_day_c`, `lk_lower_night_c` | 18.5, 17.5 | how far a room may fall in a sell |
| `lk_close_offset_c` | 0.5 | closing target above that limit |
| `lk_upper_c`, `lk_opening_margin_c` | 20.5, 0.5 | thermostat in a buy; a room closer to it is not opened |
| `shift_buy_max_gap_h` | 2 | a buy further from its sell does not open |
| `shift_closed_energy_share` | 0.6 | the pump energy left in a closed hour |
| `floor_mix_max_c` | 40 | what the shunts let through |
| `salon_open_target_c`, `salon_open_floor_c` | 20, 19 | a Netatmo target from this counts as comfort; the limit in a sell then |
| `salon_closed_floor_c` | 17 | the limit otherwise |

The add-on overrides `target_c` to 18.5 and `min_authority` to 1.0 (read 2026-11-09).

## 7. Heat demand, in theory

Heat balance per zone:

```text
C dR/dt = emitter(flow, valve) + other heat - H (R - T)
```

### Loss per zone

| Zone | H, W/K | Demand at -9.6 C and 20 C inside | Source |
|---|---|---|---|
| Ground floor, all | 266 | 7888 W, 63 W/m2 | [manual] heat demand analysis |
| of which kitchen / large hall | 72 / 33 | 2127 / 975 W | [manual] no floor loops in the LK calculation |
| of which Vardagsrum/Matsal | 85.5 | 2531 W, 70.6 W/m2 | [manual] |
| of which the Bibliotek thermostat's rooms | 76 | 2255 W | [manual] library, guest room, small hall, WC |
| Salon | 84 | 2492 W, 78 W/m2 | [estimate] U-values assumed; winter nights with open valves measured about 85 W/K |
| Upper floor, guest house | not calculated | - | no analysis exists |

### Emitters

| Emitter | Output | Source |
|---|---|---|
| Vardagsrum floor | 78 W/K of flow minus room | [manual] LK calculation |
| Bibliotek thermostat's floor | 89 W/K | [manual] LK calculation |
| Salon radiators | `2120 * ((flow - 2 - room) / 30)^1.3` W, valves open | [manual] rating; exponent and 4 K drop assumed |
| Shunts | floor flow at most 40 C | [observed], set by hand |

### Demand by outdoor temperature

Ground floor and salon are `H (20 - T)`. The whole house is the pump's own heat counter, days below 12 C, 226 days,
Oct 2025 - Nov 2026: `9.64 - 0.606 T` kW, zero at about 16 C **[observed]**.

| Outdoor C | Ground floor kW | Salon kW | Whole house, measured kW | Rest kW |
|---|---|---|---|---|
| 12 | 2.1 | 0.7 | 2.4 | -0.4 |
| 10 | 2.7 | 0.8 | 3.6 | 0.1 |
| 7 | 3.5 | 1.1 | 5.4 | 0.8 |
| 5 | 4.0 | 1.3 | 6.6 | 1.3 |
| 2 | 4.8 | 1.5 | 8.4 | 2.1 |
| 0 | 5.3 | 1.7 | 9.6 | 2.6 |
| -3 | 6.1 | 1.9 | 11.5 | 3.4 |
| -5 | 6.7 | 2.1 | 12.7 | 3.9 |
| -9.6 | 7.9 | 2.5 | 15.5, extrapolated | 5.1 |

"Rest" is the measured whole house minus the two calculated zones: upper floor and guest house, less sun, people, the stove
and the air-to-air pump in the salon, none of which is metered. Below 12 C the coldest days are few, and nothing below -4 C
has more than one day in the data.

### Flow needed for 20 C **[derived]**

| Outdoor C | Vardagsrum | Bibliotek | Salon, 250 W other heat (0-500 W) | Pump target, setpoint 18 |
|---|---|---|---|---|
| 12 | 31.0 | 29.6 | 30.7 (26.4-34.4) | 32.1 |
| 10 | 33.7 | 32.0 | 33.2 (29.4-36.7) | 33.5 |
| 7 | 37.8 | 35.6 | 36.8 (33.3-40.0) | 35.7 |
| 5 | 40.5 | 38.0 | 39.0 (35.7-42.1) | 37.1 |
| 2 | 44.7 | 41.6 | 42.2 (39.0-45.2) | 39.2 |
| 0 | 47.4 | 44.0 | 44.2 (41.2-47.1) | 40.6 |
| -5 | 54.2 | 50.0 | 49.1 (46.2-51.8) | 44.2 |

- Vardagsrum `20 + 1.37 (20 - T)` and Bibliotek `20 + 1.2 (20 - T)` come from measured winter ratios, with hot water
  taking part of the time. The floor gets at most 40 C, reached at about +5 C and +3 C outdoors; below that the rooms sit
  under 20 C unless other heat covers them.
- The salon's 250 W of other heat is measured in blocks with open valves (sd about 200 W).
- Salon steady-state only. Its dynamics are not validated: 12 h free runs barely beat "no change". **Recheck after the salon
  valve opened on 9 Nov 2026.**
- The pump target is the running curve, `40.28 - 0.710 T` plus 0.35 at setpoint 18.

### Pump capacity

**[manual]** At flow 40 C: about 12.5 kW at -7 C and 9.8 kW at -15 C. Against the measured demand, the pump alone covers the
house to about -5 C, before hot water takes its share of the time. Below that comes the electric heater, the stove or a cold
room.

### How much heat the house can store

`C = H / k`. With the ground floor's 266 W/K and a measured `k` of 0.011-0.030 per hour, C is about **9-24 kWh/K [derived]**.
The range is wide and C cannot be read directly from the data. A swing of 1 C on the ground floor is then 9-24 kWh of
heat, or about 3-7 kWh of electricity at COP 3.5, as an upper bound on one window.

## 8. Energy calculations

### Electricity by outdoor temperature **[observed]**

The pump's counters, whole days, 296 usable of 347, Oct 2025 - Nov 2026. Electricity is compressor only, as the heater counter
is unreliable. The setpoint varied over the period.

| Outdoor C (mean) | Days | Heat kWh/day | Electricity kWh/day | COP | Hot water heat / electricity kWh/day | Plan model kWh/h | Measured kWh/h |
|---|---|---|---|---|---|---|---|
| -4..0 (-1.5) | 24 | 244 | 96.5 | 2.53 | 21 / 9.0 | 3.99 | 4.02 |
| 0..3 (1.4) | 28 | 214 | 80.7 | 2.66 | 31 / 12.6 | 3.31 | 3.36 |
| 3..6 (4.6) | 36 | 174 | 57.7 | 3.02 | 29 / 11.6 | 2.55 | 2.40 |
| 6..9 (7.4) | 73 | 122 | 34.3 | 3.55 | 29 / 11.1 | 1.89 | 1.43 |
| 9..12 (10.6) | 64 | 76 | 14.5 | 5.23 | 26 / 9.6 | 1.13 | 0.60 |
| 12..15 (13.5) | 32 | 56 | 8.6 | 6.46 | 24 / 8.3 | 0.45 | 0.36 |

The COP comes from the pump's own counters and is not independently measured. Datasheet **[manual]** at flow 35 / 45 C: 4.72 /
3.61 at +7 C, 4.03 / 3.20 at +2 C, 2.69 / 2.32 at -7 C.

### What the plan assumes

```text
flow target  = 40.28 - 0.710 T + what the setpoint adds
compressor   = -1.026 - 0.236 T + 0.259 setpoint   kWh/h
heater       = 0.206 * max(flow target - 44, 0)    kWh/h
```

Values fitted 2026-11-09. "Plan model" above is this at setpoint 18. It is fitted on pure-heating hours and measured counts all
hours including hot water, so measured should be lower by the hot-water share. It matches below about 5 C and is too high when
mild: 1.89 against 1.43 kWh/h at 7 C. The plan therefore overstates what mild-weather sells can save.

**What the setpoint adds:** measured per setpoint in the fit, in C against the curve: 15: -8.4, 16: -5.6, 17: -3.2, 18: +0.3,
18.5: +1.6, 20: +6.5, 21.5: +10.9. That is about 3 C of flow per degree of setpoint, which agrees with the last 40 days
(2.99 C per degree, RMSE 0.54 C). The fixed `flow_per_rts` of 1.9 is used only outside the measured setpoints.

### The forecast of the electricity (built, shown, not yet used by the plan)

A walk-forward test over four years compared forecasts issued at 14:00 the day before ([study and review](../documentation/energy-forecast/README.md)).
The chosen model (`plan/energy_forecast.py`) has two parts:

```text
heating = ridge on outdoor hinges, trailing outdoor, estimated flow, sun and wind, times a 3-day level correction
hot water = mean per clock hour over 28 days, weekday and weekend apart
```

On the held-out 227 days it misses a day by 3.8 kWh (6.7 %) with measured weather and 4.6 kWh with a next-day weather forecast,
against 8.3 kWh for the plan's energy model above and 8.4-9.9 kWh for BESS's two forecasts. The flow target can be computed to
about 0.5 C a day ahead, but flow alone gives 14 kWh/day of error: it matters together with the outdoor temperature, because
how many loops are open decides the heat drawn.

It is good enough to plan with. It cannot yet measure savings: the setpoint response is unvalidated, and monthly savings below
about 6 % are inside its error.

### What one degree costs

| Change | Effect | Source |
|---|---|---|
| Whole house one degree warmer | 606 W/K = 14.5 kWh heat/day = about 4 kWh electricity/day at COP 3.5, about 12 % at 7 C | [derived]; the savings study got 14 % |
| Setpoint one degree, as it happened | below 18: 0.23-0.25 kWh/h; above 18: 0.35-0.37 kWh/h, +/-0.04 | [observed], 11 Oct - 9 Nov, 3-14 C, loops as they happened to be |
| Flow +1 C with the same heat delivered | about 2.4 % worse COP at 7 C | [manual], datasheet 35 to 45 C |
| Flow +3 C (one setpoint degree), other rooms throttled | about 7 % of 34 kWh/day, 2-3 kWh/day | [derived] |
| Salon one degree warmer | 84 W/K = 2 kWh heat/day = about 0.6 kWh electricity/day | [derived] |

### What closing the floor loops is worth

The rooms Theria can close draw about 167 W/K of emitter at most: at 8 C outdoors, flow 34 C and room 19 C, that is about 2.5 kW
with all loops open. Since 5 Nov 2026 about 0.42 of the loops are open, so about 1.05 kW of heat, or **0.3 kWh/h of
electricity at COP 3.5 [derived]**. The regressions measured 0.1-0.3 kWh/h **[observed]**, so the calculation and the
measurement agree. Kitchen and large hall have no floor loops, and the upper floor, salon and guest house keep drawing.

The closed-hour energy in the optimizer (`closed_energy_share` 0.15) is a guess; closed hours measured about 0.6 of open.

### Hot water

24-31 kWh of heat and 8-13 kWh of electricity per day, COP 2.4-2.9, about 15-20 % of the pump's time. Theria does not control it.

### The electric heater

The heater can start after a hot-water block, when the pump's degree-minute deficit has built up, at flow targets of 31-39 C
as well as above 44 C. Five starts since 1 Oct 2026 out of 331 hot-water blocks, all since 4 Nov. The plan's model only
covers flow above 44 C, so it does not see these starts. `sensor.aux_cons_heating` is stuck, so this energy is unmeasured.

## 9. How the saving is calculated

**Agreed definition (2026-11-09), not yet built:** per hour and summed, the cost of the energy use without control minus the real
cost. "Without control" is the forecast at the base setpoint and normal thermostats, using the outdoor temperature that actually
occurred, times the hour's price. It must not be fitted on hours when the plan was shifting; control days check it.

**Built today (`outcomes.py`):** measured compressor energy scaled by the model's ratio between the base and the actual setpoint,
at the hour's price. Hot-water hours are left out. Heater energy counts against the plan in hours when the setpoint was raised.

**What the numbers say so far:** a shift with the two measured slopes earns a median of 0.69 kr/day on 44 days, p10 -0.34.
Closing the loops is worth about 0.1-0.5 kr per mild day, inside the noise. Neither is measured on real controlled days.

## 10. What the models do not capture

- Sun, the stove, people and the salon's air-to-air pump are unmetered. A fire can change a room by up to 4 C.
- The room cannot be predicted hours ahead better than "no change"; the model is used only for the effect of a change.
- Nothing below about 3 C has been measured with the loops closed, and the rebound after closing is unmeasured.
- The Bibliotek thermostat closes rooms that its sensor does not see.
- Upper floor and guest house have no heat demand calculation.
