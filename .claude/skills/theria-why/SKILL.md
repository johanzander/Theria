---
name: theria-why
description: Use when Johan asks why Theria did or did not do something (a setpoint, a plan hour, a cut that did not happen, a thermostat write, a saving figure, a chart line). Answers only from evidence, never from guesses, and says what data is missing when it cannot.
---

# Why did Theria do that?

An explanation is a claim with proof. If you cannot show the data that makes the claim true, you do not
have an explanation yet. Never say "probably", "most likely", "it looks like" or "I suspect" as the
answer. A suspicion is a lead to check, not a finding.

## The rule

Every answer uses these three headings, written out, in this order (no heading may be skipped; an
empty one says "the data does not show ..."):

1. **What happened** - the observed fact, with its timestamp and whether the time is UTC or local
   (Theria's tables and logs are UTC; the plan table and chart are mostly local - check which).
2. **Why** - the cause, stated once, as a fact.
3. **Evidence** - the rows, values and settings that prove it: quote them (time, field, value).
   A reader must be able to check the answer without trusting you.

If a part cannot be filled from data, say exactly that: "the data does not show X; Y is missing".
Then do the second half of this skill (close the gap), do not guess.

## Where the evidence is

Read these before concluding. Use the running add-on, not only the code.

| Question | Source | How |
|---|---|---|
| What did the pump/thermostat actually have? | Execution trace (`pump read`, `pump result`, `thermostats read`) | debug export `GET <ingress>/api/export-debug-data`, section "Execution trace" |
| What was written, refused, paused? | Action log ("What it did" on the Control tab) | debug export "Action log"; times are UTC |
| What did the plan choose per hour, and its reason code? | `GET api/plan/day?day=YYYY-MM-DD` (`rts_planned`, `rts_measured`, `reason`, `price_sek_per_kwh`, `floor_c`, `extreme`) | `hours[].start` is UTC |
| What inputs/settings was a plan computed from? | Debug export "Plan runs" and "Settings" | replay the plan with `core/theria/plan/planner.py` on those inputs |
| Why does a reason code mean what it says? | `core/theria/plan/planner.py` (reason strings) and `documentation/debugging-plans.md` | read the code that sets it |
| What was a sensor/entity at a time? | Home Assistant history, InfluxDB | `/api/history/period/<start>?filter_entity_id=...` |
| Radiators/salon/guest house | `GET api/radiators/status`, `api/radiators/log` | shadow only: nothing is written |

Ingress URL: open the add-on in Chrome (`https://casazander.se/local_theria`), find the iframe `src`
(it holds the ingress token) and fetch from the iframe's `contentWindow` so the session is reused.

## Method

1. Pin down the exact claim: which quantity, which hour, UTC or local, what Johan expected instead.
2. Pull the **actual** value first (execution trace, history), then the **planned** value (plan day).
   They differ more often than you expect: the plan may say 16 while the pump held 18. Explain each
   difference separately; never explain the chart with the plan's logic when the pump did something else.
3. For a plan choice: replay the planner on the stored inputs for that run and show the cost of the
   alternatives (the chosen action and its neighbours). "It chose X" is not why; "X cost a, Y cost b,
   the binding limit was c" is why.
4. Check the obvious blockers by name and report each as ruled in or out with its evidence: the
   authority limit (`min_authority`, effective range), the ramp limit (`max_rts_step_per_hour_c`),
   comfort floors/ceiling and night window, extreme-price floor, safety/trouble pause, hand-change
   pause, executor refusal ("read back"), mode (off/shadow/active).
5. Rule each blocker in or out in the answer with its evidence, including a failed write
   ("read back" = the pump still reported the old value after the write; the cause of that is not in
   the trace until it is recorded). Compare with the neighbours (same price, same state) - a one-hour exception needs a cause that
   differs in that hour.
6. State the conclusion and the evidence. If two causes both fit the data, say so and name the
   measurement that separates them. Do not pick one.

## Closing a data gap

When the data cannot separate the causes, the answer is "not provable yet", followed by the change that
makes it provable. Add the missing record, then wait for it to occur, or replay from stored inputs.
Never ship an explanation that the system cannot reproduce.

Known gaps (update this list when one is closed):

- **Why the planner chose an action - closed 2026-11-04 for plans computed after that deploy.** Each hour
  carries `alternatives`: every setpoint the plan could pick, with `extra_objective_sek` (what it would
  have cost over the whole horizon against the chosen one; positive = worse) or `excluded` (`flow_cap`,
  `ramp_limit`), and for the first hour `held_by_commitment` (the applied value was kept against a better
  one: held, best, gain, margin). Read it from `GET api/plan/day` (`hours[].decision`) or from the debug
  export's "Plan runs" (`decided[].alternatives`). Runs and hours from before the deploy have none: replay
  the stored run (`scripts/replay_plan.py`, `core/theria/plan/plan_run.replay`) and read the alternatives
  from the replay. A winner whose neighbours are within a few öre is a near tie, and the answer says so.
- **Which input made the applied value differ from the plan:** the trace has the asked and the result,
  but not the constraint that changed the ask (pause, trouble, executor limit). Record the reason for
  every deviation next to the ask.
- **Why a write failed `read back X, wrote Y`:** the trace shows the pump still read X after the write
  (2026-11-04 16:12 UTC, asked 18.0, pump 20.0), not whether the pump refused, was slow to report, or
  was changed by something else. Record the read-back delay and the value read at each retry.
- **Event times:** keep UTC in storage; label the zone wherever a time is shown.

## Do not

- Do not answer from reading the code alone when the running system has the record.
- Do not call something "the cause" because it is a known issue; show it was active at that time.
- Do not use rounded or remembered numbers; quote the values.
- Do not change settings or write to the pump while investigating, unless Johan asks.
