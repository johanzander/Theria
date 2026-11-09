# Part 1: floor heating, thermostats and loops per room

Status: built on 2026-11-09, except a lead before a sell (none: 0 h). Johan chose to go to active mode without the two-week test in section 7. Part of
`2026-11-09-closed-loop-shift-design.md`.

## 1. Purpose and limits

During a sell, each LK room's loops are closed by its thermostat. During a buy just before a sell, the thermostat is set
above the room so that the loops are open and the floor can store the heat. Johan decided on 2026-11-09 that the floor
loops may be closed and that a sell also lowers the pump setpoint.

In scope: the two LK thermostats with sensors. Vardagsrum: 3 loops. Bibliotek: 5 loops, which also serve the guest room (2 loops), the small hall and the WC (Johan); the sensor stands only in the library, so those rooms are closed unobserved.
Out of scope: the pump setpoint (part 2), the salon (part 3), how windows are found (part 4), measuring (part 5), the
forecast (part 6). The upper floor has no remote control and is not closed.

Part 4 delivers buy and sell windows, fixed when calculated. This part returns, per room and per 15 minutes, the
thermostat target.

## 2. What exists

- `lk_executor.py`: writes thermostats, reads back, rewrites whenever the read value differs, keeps late values as pending,
  pauses 1 h on a hand change, restores `lk_base_c` when leaving active mode. Outside a hold it writes `lk_limit_c`. The
  pause and the read are for all thermostats at once.
- `service._plan_valve_holds`: holds every thermostat at the hour's `floor_c` when the plan's hour has `valves_closed`.
  `floor_c` is `extreme_floor_c` (17) in extreme-price hours.
- `heat_shift.py` never sets `valves_closed`; a test requires it.
- `regulator.py`: learns how far a room rises after closing, only for closures near 22. Inactive.

## 3. The algorithm

Per room, every 15 minutes:

1. Read the room temperature, the share of its loops open and the thermostat target.
2. Mode, in this order of precedence:
   - **closing:** from the start of a sell minus the lead, to the end of the sell;
   - **opening:** during a buy, if the buy ends at most `buy_max_gap` before its sell starts, and the room is more than
     `opening_margin` under the upper limit. A buy further from its sell leaves the thermostat alone: the heat is gone
     before the sell, and only the pump moves;
   - **normal** otherwise.
3. Thermostat target:
   - closing: the room's closing target (section 4), from its day or night limit;
   - opening: the room's upper limit;
   - normal: the normal target.
4. Ask the executor for the target. It keeps rewriting while the value read differs, as today.
5. A hand change on a thermostat pauses that room only, and ends the window for that room. After the pause Theria does
   not overwrite the value until the next window.
6. When a guard ends a sell (part 3 for the salon, a sensor gone) only the pump returns to base. The thermostats stay
   closing to the end of the window, so that a room near its limit is not opened at the dearest price.

Rules that follow:

- A window is committed when its lead starts, not when it first affects the pump, and it does not move after that.
- The lead counts as part of the window for the cooling limit in part 4.
- Part 4 drops buy hours inside a sell's lead.
- The thermostat is the per-room guard during a sell; the mean-room guard in `shift_rules.py` is removed.

Changes in the code:

- A separate sell and buy flag per hour; today's `adjust < 0` cannot express it. `valves_closed` follows the sell flag.
- The executor reads and pauses per room, and its value outside a hold is `lk_base_c` in the heat-shift strategy. The
  optimizer path keeps `lk_limit_c`.
- Part 1 computes its own day/night limit and does not use `floor_c`.
- A pending late value expires after the grace period, and one arriving late is not taken for a hand change.
- Plan energy for closed hours is not `closed_energy_share` 0.15; the closed hours measured about 0.6 of open (part 6).
- The room forecast and disturbance use the normal target instead of `lk_limit_c` 22, and closed hours in `next_delta_c`
  use the closed-loop curve.

## 4. Settings

| Setting | Value | Note |
|---|---|---|
| Normal target | 19.5 (`lk_base_c`) | exists |
| Lower limit, day / night | 18.5 / 17.5 | |
| `lk_close_offset_c`, closing target above the lower limit | 0.5 | closing target = lower limit + this; a room reopens fully about this far below its target |
| Upper limit in a buy | 20.5 | |
| `opening_margin` | 0.5 | a room closer to the limit is only half open |
| `buy_max_gap` | 2 h | 0.3 C of stored heat lasts about 2.5 h at 0.12 C/h |
| Lead before a sell | 0 or 1 h, tested | |
| Grace period for late values | 1 h | |

## 5. What the data says

29 Oct - 8 Nov 2026, HA actuators, outdoor about 3-13 C. Reviewed and reproduced.

| Question | Result |
|---|---|
| Electricity saved by closing every ground-floor loop | **0.1-0.3 kWh/h, not settled.** 0.28-0.34 without the room temperature in the regression; 0.09-0.20 (+/-0.12) with it (checked). Loop state follows the room, and warm rooms also mean sun and stove days, so neither is clean |
| Worth per day, mild weather | about 0.1-0.5 kr, inside the noise |
| Setpoint in the same regression | 0.14-0.20 kWh/h per degree |
| How open the loops already are | 5-8 Nov, all targets 19.5: 15:00 0.21, 16:00 0.27, 17:00 0.32, 18-19:00 0.43, 00-05 0.30-0.41 |
| Target against room, share open | under 0.5 C above the room: 0.18-0.56; at least 0.5 C above: about 0.9. A room reopens fully about 0.5 C below its target |
| Time to close after lowering the target | median about 1 h, but 20 of 26 events are drops from 22 and only 2 from 19.5. Not representative of a sell |
| Time to open after raising the target | at most 0.5 h (16 events) |
| Closed-loop cooling, dark, about 8 C | Vardagsrum 0.115 C/h (n 4), Bibliotek 0.123 C/h (n 5). Near 0 C older data gives up to 3 times more, so part 4 must not use a straight line from these |

**A buy far from its sell does not reach it.** After a buy 00-05 the room is warmer and stays closed about 2.5 h longer, so
the heat is spent by 08. For 15 of 55 sells the buy ended 8-12 h earlier. The closed and open share also means a thermostat raise can add more energy than the
sell removes (about 1 kWh against 0.5). Hence the gap condition.

## 6. Risks

- **Water volume for defrost**: accepted by Johan; the buffer tank is unrecorded and the upper floor stays open. Watch for
  defrost problems and alarms in the first runs.
- **A crash during a window** leaves the targets where they are until the add-on runs again: a night's closing target
  stands into the day, a buy leaves the loops open. Theria writes the normal target at start.
- **Leaving active mode with a failed restore** stays active, as today; the next cycle must not reapply a window.
- **Rebound**: unmeasured.

## 7. For Johan to decide

1. Decided by Johan 2026-11-09, yes: test first, two weeks with a closed-loop sell on random days against normal days, with no buy and no lead, read per
   your definition of saving. It decides whether the rest is worth building. Recommended before the full build. The
   review also recommended keeping buys out until that test; I have kept them, with the gap condition, as you want
   consumption moved.

## 8. Tests

Unit: precedence (closing over opening), window committed at lead start, the gap and margin conditions, day and night
limit, guards leaving the thermostats closing, per-room pause and read, late value expiry. Service: a sell closes in
active mode and writes nothing in shadow.

## 9. Independent review (2026-11-09)

Ten findings; all are taken into the above except the recommendation to leave buys out, which is Johan's call. Verdict on
the previous version: not to build as designed; the per-room thermostat as a self-limiting guard is worth keeping.
