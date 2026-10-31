# Debugging a plan

The plan is a pure function of its inputs, the settings it ran on and the model parameters. The
add-on therefore keeps **every plan computation** (every 15 minutes, 7 days) with all three in
`/data/plan_runs.jsonl` (`.plan_runs.jsonl` locally), the way BESS Manager keeps a prediction
snapshot per optimisation run. `plan_record.jsonl` only keeps the first row of each hour (for the
charts) and is not enough to reproduce a plan.

Fetch the runs (the API is behind ingress, so from a browser session on Home Assistant):

    fetch(<ingress url> + "api/plan/runs?hours=6").then(r => r.text())

Save the JSON as `runs.json`, then:

    python scripts/replay_plan.py runs.json --list
    python scripts/replay_plan.py runs.json --at 2026-10-31T16:00
    python scripts/replay_plan.py runs.json --at 2026-10-31T16:00 --set comfort_cost_sek_per_c_hour=2

The replay prints whether it decides what the run decided at the time (it should: if not, the
planner changed since), the hour table with the binding reason, and, with `--set`, the plan with
a changed `PlanSettings` field next to it. Times are UTC; the UI shows local time.

Not kept: the weather forecast and the sensor readings themselves, only what the planner was fed
(`PlanInputs`: prices, outdoor, sun, extra drift, starting room and slab).

## What keeps the plan from flipping (2026-10-31)

On 2026-10-31 the first action alternated between 16 and 18/20 from one replan to the next (pump
writes 18, 16, 18, 16, 18, 16 within three hours), once in the dearest hours. An external review of
`planner.py` verified these causes (the replays use the fitted parameters; the exact state of that
day was not saved):

- A radiator calling for heat raised `rts_min` to the base inside the optimisation, which turned
  "valves closed at 16" into "valves closed at 18". It is now applied when the setpoint is written
  (`PlanExecutor.execute(floor_c=...)`) and counted for an hour after the last call.
- The authority-scaled setpoint range changed the action set between replans; it is fixed per day.
- The disturbance estimator divided a 0.1 C sensor step by 0.25 h (0.4 C/h of imagined drift) and
  the planner sits on a knife edge between "all closed" and "heat everything" (a drift change of
  0.01-0.02 C/h crosses it). It now judges over at least an hour, reaches 2 h back and is capped at
  0.15 C/h.
- The first hour now keeps what was applied (setpoint and valve state) unless another choice is
  better by more than `commit_margin_sek` (0.3 kr; `PlanInputs.previous_closed`).

Not done, from the same review: the first step is priced as a full hour at the hour's price even
when the replan runs at :48, the sun swing is the one from the hour's start, the slab is estimated
from the written setpoint and ignores closed hours, and the end-of-horizon cost is about
100 kr/C (127 h of drift). Also `min_saving_sek` replaces the whole plan by the flat setpoint when
the saving crosses 0.2 kr, which is another place where a near tie flips the first action.
