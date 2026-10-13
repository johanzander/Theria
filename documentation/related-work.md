# Related work

Other thermostat and price-control solutions we compared Theria against: what we took from them and what we
left out.

## PowerSaver, Heat Capacitor strategy (inspired by)

<https://powersaver.no/nodes/ps-strategy-heat-capacitor.html>

Uses the building as a heat store: the setpoint is raised when electricity is cheap and lowered when it is
expensive, and the climate entity is never switched off. Each cheap hour ("buy") is paired with a later expensive
hour ("sell"), and pairs whose price difference is below `minSavings` are dropped. The thermal model is two numbers
you enter by hand (minutes per degree up and down) plus a maximum offset and a short boost.

- **Taken:** the heat-capacitor idea, and the minimum-saving threshold (`min_saving_sek`: a plan that does not beat
  the flat setpoint by that much is not run).
- **Left out:** pairing of hours and hand-entered minutes per degree. Theria plans the whole day with dynamic
  programming over room and floor-slab temperature, with a pump energy model and a room model learned from the
  house's own history, comfort floor/target by time of day, and measured savings against a flat-setpoint baseline.
  PowerSaver has no knowledge of the pump's energy per degree and no feedback from the result.

## node-red-smart-thermostat (compared, not used)

<https://github.com/WojRep/node-red-smart-thermostat>

A PID thermostat for Node-RED: learns the room's time constant during the first hour (Cohen-Coon), then runs PID
with schedule, boost and away modes and a limit on change per cycle. It has no price input, no outdoor-temperature
model and no pre-heating, so it solves a different problem (holding a temperature, not moving heat in time).

- **Taken:** the idea of a bounded change per cycle and a learned correction from how the room responds; Theria's
  stage clamp, rate limit and the observer/trimming loop do this on top of a price plan.
- **Left out:** PID on the room temperature. The room responds over days (time constant about five days), so a PID
  loop would only chase the weather.
