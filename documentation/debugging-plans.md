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
