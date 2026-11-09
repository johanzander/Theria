# Closed-loop heat shift: design

Status: revised after independent review, for Johan (2026-11-09). This is the plan: nothing further is built until
he has approved it. Already written and running in shadow only: the rules in `shift_rules.py` (section 8). Nothing is
committed or deployed.

## 1. Intent

Johan's proposal, in his order:

1. A theoretical curve per zone: how it heats and cools at each indoor/outdoor difference.
2. Rules for how far the temperature may deviate.
3. A forecast of the energy use as far ahead as there are prices.
4. Scenarios: how much energy can be moved, what it saves, whether it stays inside the rules.
5. Pick a scenario and measure the outcome.
6. Repeat every 15 minutes, so a wrong forecast is corrected within 15 minutes.
7. Outside the loop, measure the errors and adapt the curves on a slower cycle. Rules are proposed, never changed
   automatically.

Why: how long and how much to lower depends on how the house responds, from 15 minutes to several hours, and cooling
in the salon and the main residence decides it. Anti-goal: a plan a person can see is absurd. The only reasonable
outcome Johan accepts is less consumption in the dearest periods.

The algorithm (section 3) is kept apart from the temperatures (section 4), which are settings.

## 2. Who controls what

| Part | What it decides | Today | Can Theria set it? |
|---|---|---|---|
| Pump room setpoint | flow temperature, for all circuits | by hand | yes |
| LK thermostats, ground floor, one per room | whether that room's loops are open: target against room | by hand | yes, slowly (cloud; a write is often confirmed late) |
| Shunts on the floors | cap the floor flow at 40 C | fixed | no |
| Upper floor | floor heating, own shunt, LK IC2 thermostats, no sensor or remote | by hand at the thermostat | no, and not observed |
| Netatmo, salon | valve opening, schedule "Mitt schema": target differs by hour and weekday | Netatmo app | shadow only, never written |
| Tado, guest house | valve opening, own targets | Tado | shadow only, never written |
| Hot water | takes the pump about 20 % of the time | pump | no |
| Tile stove, sun, people | heat Theria does not see, up to about 4 C | - | no |

What follows from it:

- **Setpoint alone cannot stop the consumption.** It lowers the flow; open loops keep drawing heat. On 2026-10-29 the
  compressor kept running with the setpoint at its minimum. Closing the loops is what stops the draw.
- **The thermostat is the stronger lever on the ground floor**: going from about 30 % to all loops open is roughly 6
  times the effect of one setpoint degree (review, reasoned from measured links).
- **A higher flow does not open a thermostat.** Heat only reaches rooms whose valves are open.
- **Every zone but the ground floor is outside Theria's reach today.** A lower setpoint reaches the salon, the upper
  floor and the guest house whatever their thermostats do.

## 3. The algorithm

Two levers: the pump setpoint (one, shared) and the thermostat per LK room.

**Per day, when prices arrive**, and again if a guard or a deviation requires it:

1. Find the dear windows from the prices (PowerSaver's sliding mean and turning points). A window must be worth more
   than `min_gain` against the hours it is compared with.
2. Limit each window by the house: for each LK room, the time the room takes to fall from where it is predicted to be
   to its lower limit at the forecast outdoor temperature, **with its loops closed**. The shortest room decides. This
   is what makes a window 15 minutes or several hours.
3. Check the forecast energy: a window cannot save more than the house uses in it.

**Every 15 minutes:**

4. Read each room and whether its loops are open.
5. Decide the mode for the hour: normal, buy or sell. A window that has started runs to its end unless a guard ends
   it, and stays ended.
6. Set the levers:
   - **Normal:** pump at base, each thermostat at its normal value.
   - **Sell:** thermostat at the room's lower limit, so that the loops close and reopen by themselves if the room falls
     to it. This works per room and keeps working if Theria stops. The pump goes below base as well: the lower flow is
     what reduces the compressor for everything Theria cannot close (upper floor, salon, guest house).
   - **Buy, in the cheap hours before a sell:** pump above base and each thermostat above the room, at the room's upper
     limit, so that the loops are open and the floor stores the heat. A raised setpoint alone stores nothing in a room
     whose loops are closed.
7. **Guards**, measured: a sensor gone, a hand change, or the salon under its limit ends the sell.
8. Compare each room with its prediction. A deviation above the threshold shortens or ends the sell on the ground floor.

**After a sell** the thermostat returns to normal and the loops open: the room takes the heat back. Part of the
consumption moves there, so what a buy before the sell saves against that rebound has to be measured.

**The saving** (Johan, 2026-11-09), per hour and summed: the cost of the energy use without control (the forecast from step 3, at the base setpoint and normal thermostats, using the outdoor temperature that actually occurred, times the hour's price), minus the real cost (measured electricity times the same price). Nothing else counts as the saving. Control days check that the no-control estimate is right; it must not be fitted on hours when the plan was shifting.

**Slow loop, outside the algorithm:** per sell, log what was closed, flow, compressor electricity during and after, and
each room. Weekly, fit the closed-loop cooling per room and the electricity saved and paid back, and store them as a
proposal with their uncertainty, applied on Johan's approval. Rules are never loosened automatically. About one day in
seven stays normal, picked at random, as a control.

**Added later only if something can differ:** scenario lists: while
depth is a fixed step and saving is linear in it, a list has only one winner (review, finding 9).

## 4. Settings

| Setting | Value now | Note |
|---|---|---|
| Normal thermostat target, LK | 19.5 (`lk_base_c`) | as set by hand |
| Upper limit per room during a buy | 20.5 | normal + 1 |
| Lower limit per room, day / night | 18.5 / 17.5 (`floor_c`, `night_floor_c`) | |
| Salon target / limit, comfort hours / otherwise | 20 / 19 / 17 | Theria's values; Netatmo's own schedule says 21 |
| Setpoint step | 1, at most 2 (`shift_rts_c`) | |
| Smallest price difference | 0.5 kr (`shift_min_gain_sek_per_kwh`) | |
| Deviation threshold | 0.5 C, ground floor only | was 0.3, inside the noise |
| Lowest outdoor temperature for selling | 3 C | lowest measured; lowered as closed-loop runs are measured |

## 5. Data

History 11 Oct - 9 Nov 2026, outdoor 3.3-14.6 C, unless stated.

| Question | Result |
|---|---|
| Electricity per setpoint degree | **below 18: 0.23-0.25 kWh/h, above 18: 0.35-0.37**, +/-0.04 (own check of the review, block lengths 3, 4, 6 h, with and without time of day; only 13-25 blocks above 18). The earlier 0.29 is the mean of the two |
| Saving of the present plan, 44 days, 5 h windows | at 0.29 both ways median 2.06 kr/day. **With the two slopes 0.69 kr/day, p10 -0.34.** The 2 kr figure given earlier does not hold. These slopes come from setpoint changes with the thermostats where they happened to be, so they say nothing about a raise with the loops opened for it |
| Sell only, setpoint -1, heat paid back at the day's mean price | median 0.9-1.1 kr/day, p10 0.2-0.3 |
| Closing the loops | energy saved not measured. `closed_energy_share` 0.15 is a guess, and the upper floor, salon and guest house keep drawing |
| Closed-loop cooling, ground floor | 0.09-0.27 C/h at 7-11 C, median 0.13 |
| What opens floor loops | target against room, not flow. Target 0.5 C above room: 96-98 % open at any flow. 0.5 C below: 9-12 % |
| Loops open by thermostat regime | **30 Oct - 4 Nov, thermostats moved around: 0.71. Since 5 Nov, fixed 19-19.5: 0.42**; 15: 0.21, 16: 0.27, 17: 0.32 |
| Setpoint to flow | 1.9 C per degree as a mean; **17 to 18 is about 3.6 C, 18 to 19 about 2** |
| Response of the electricity | within the first hour; no rebound seen in 1-12 h, with loops not closed |
| Baseline electricity at setpoint 18 | measured 1.36 / 0.98 / 0.69 kWh/h at 5 / 9 / 12 C; the fitted model says 2.45 / 1.51 / 0.80 |
| Setpoint chatter | 344 changes in 29 days, median hold 45 min |
| Floor heat to room with loop state known | gain 0.0058-0.0078 per h (curve 0.0065); loss 0.0083-0.0139 (curve 0.016) |
| Noise of that fit, 3 h | night 0.08-0.09 C, all hours 0.20-0.23 C |
| Salon | no constant curve fits: 8-9 Nov rose 0.75 C in 8 h (curve: fell 0.2); 13 winter nights -0.4 to +0.3 C at the same demand. Flow-limited while open: setpoint 18 gave 20.3-20.6, 17 gave 18.4-19.5 |
| History of loop state | LK and Netatmo thermostats are in InfluxDB, `binary_sensor.lk_*_actuator_*` is not (about 10 days in HA) |

Not measured: closing the loops in a sell (saving, cooling per room, the rebound after it); anything below 3 C; the
pump's behaviour with few loops open.

## 6. Independent review (2026-11-09), and what was done with it

| # | Finding | Disposition |
|---|---|---|
| 1 | Energy slope is not symmetric; saving falls to 0-1 kr/day | Verified, section 5 |
| 2 | The loop table mixed thermostat regimes; 0.60 is 0.42 now | Verified, section 5 |
| 3 | The thermostat is much the larger lever; the design did not treat it as one | Accepted; it is now a lever in the algorithm |
| 4 | The salon limit ended most morning sells as the room warms up to a new target | Accepted, not solved: see 7.3 |
| 5 | The "delivered heat" formula was physically wrong | Removed |
| 6 | No cold limit; a January buy never reaches the floor past 40 C | Cold limit added; a raise must open the loops, and has no effect on the floor above 40 C |
| 7 | Upper floor, guest house and stove were missing | Added to section 2 |
| 8 | The formula's inputs do not exist for future hours | Formula removed |
| 9 | Scenarios and R1 never decide anything | Scenarios removed until something can differ; the cooling limit now binds |
| 10 | The 0.3 C trigger sits in the noise; extending on warm days harms the salon | 0.5 C, ground floor only, no extending |
| 11 | Executor order, late LK confirmations, crash leaves values | Open for the build: section 8 |
| 12 | Simplest version: sell only; hot water may be a larger lever | Not adopted: Johan's intent is to move consumption, so buys stay. Hot water to be checked |

Verdict of the review on the previous version: not to go live.

## 7. For Johan to decide

1. Decided by Johan 2026-11-09: the floor loops may be closed. Whether the house has a buffer tank remains unrecorded (the manual warns about water volume for defrost with all floor loops closed; the upper floor's loops are not closed by Theria).
2. Decided by Johan 2026-11-09: a sell lowers the pump setpoint and closes the loops. Lowering also reaches the salon, which is flow-limited while open (question 3).
3. **The salon's limit in a sell.** What counts is unclear while the room is warming up to a new Netatmo target; simplest
   is to check it only during 15-19.
4. Netatmo Komfort 21, or 20 as set in Theria.
5. Control days.

## 8. Stages

1. Write the actuators into InfluxDB (Johan's Home Assistant).
2. Done 2026-11-09: the salon limit follows the Netatmo target.
3. Buys and sells with the thermostat as second lever, ordering of the writes and handling of late LK confirmations.
4. A measured test on real days, with Johan's go: a sell with closed loops and a lower setpoint, with and without a buy
   before it. Read the saving, the cooling and warming per room and the rebound.
5. Only then: window length from the measured cooling, the energy limit (baseline refit), deviation handling.
6. Log per window, control days, weekly proposals.
