# A buy only when a zone can store the heat

Status: built 2026-11-09 after independent review, which replaced the first version of section 3. Not deployed. Amends `2026-11-09-closed-loop-shift-design.md` and part 1.

## 1. The gap

The heat shift picks buy hours from the price alone. It never asks whether anything can take the heat.

Johan's case, 9 Nov 2026: the tile stove had warmed the ground floor (Vardagsrum 22.7 C, Bibliotek 20.1 C, thermostats at 19.5 C, all six
actuators closed). The plan still raised the setpoint 23-04. The room was forecast to fall all night with or without the raise, and
the forecast did not reach 19.5 C until 09:00.

Replay of that plan with the plan's own model (`cost` against the flat setpoint):

| Option | Cost, kr | Ground floor at 14:00, loop state counted, C | Salon at 14:00, C |
|---|---|---|---|
| Sell only | -22.2 | 0.00 | 0.00 |
| +0.5 from 21 to 04 | -19.3 | 0.00 | +0.03 |
| Plan, +1 from 23 to 04 | -18.2 | 0.00 | +0.04 |
| +1 from 11 to 14 | -18.3 | +0.02 | +0.03 |
| +0.5 from 04 to 14 | -16.0 | +0.01 | +0.05 |

**Moving heat to cheaper hours is not a goal in itself.** It pays only if the heat is stored in a zone that would otherwise have bought it
at the dear price. A house warm enough to coast through the sell needs no buy before it.

## 2. What actually stores heat

A higher flow only delivers more heat through a valve that is open, and a thermostat that is satisfied closes it.

| Zone | Stores a raise for the sell? |
|---|---|
| Ground floor, per LK thermostat | only with the loops open. Opening mode raises the target for this purpose |
| Salon | no. A raise moves it +0.03-0.05 C, below the sensors' 0.1 C. Its memory is long but the amplitude is too small |
| Guest house, upper floor | no: about 60 % of the house's 606 W/K by difference, so a raise is real heat there, but it is never stored for a sell |

A raise with closed loops still costs electricity. How much is unmeasured: only about 5 hours of data exist.

## 3. The rule

The question is not whether a zone can take heat, but whether the heat will be needed at the sell. Order of work in `plan_shift`, after `balanced`:

1. Forecast each room through the sell without the buy: the flat room, the delta of the sell with closed loops, and each thermostat's offset from the mean now.
2. A buy run stands only if it ends at most `shift_buy_max_gap_h` before a sell, and some room both ends that sell under its closing target and can be opened at
   the buy (`thermostat_target` gives `lk_upper_c`).
3. Each buy hour must also reach the floor: above the 40 C shunt cap the raise is dropped.
4. A sell with no buy left must beat the price the heat is bought back at: mean price of the sell, minus the mean of the same number of hours after it, above
   `shift_min_gain_sek_per_kwh`.

One test for all hours from the forecast. There is no measured path at the first hour, no salon reason to buy and no new guard, so a buy cannot appear
hour by hour. Without loop control (`plan_closes_valves` false) the rule is off.

**Consequence: buys become rare.** The cheapest hours are usually far from the sell. Most days are a sell with nothing before it. That is what the data says a
far buy is worth.

## 4. The forecast

- Ground-floor delta during a raise is scaled by the loop opening: `gain * open * flow_delta`, and the relaxation uses `gain * open + loss`. Today it
  ignores the loops, so a chart shows a gain that cannot happen.
- The chart's "limit" line follows the thermostat target actually in use (19.5 C in the heat shift, not 22 C).

## 5. Also changed

- Closing the loops saves energy only in proportion to how open they would have been: `1 - (1 - 0.6) * open share`. A warm house shows no closing gain.

## 6. Limits

- The plan's flat energy is about twice the counter in mild weather (2.1 against 1.1-1.2 kWh/h at 7 C), so the modelled kronor are too high. A reviewer's estimate of a
  5 h sell is about 13 kr at most, against the 22 kr modelled.
- The valve ramp in the room model was too narrow. Refitted 2026-11-10: loops start to open 0.75 C above the target, all open 0.95 C below it.
- Raises with closed loops and the rebound after a sell are essentially unmeasured.
- The forecast on a stove day was 1.1 C too low at 8-12 h, so the rule can be wrong on such days. The plan is recomputed every 15 minutes from the measured room.
