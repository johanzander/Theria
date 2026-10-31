#!/usr/bin/env python3
"""Replay a plan the add-on computed, from the inputs it kept, and take it apart.

The add-on keeps every plan computation with its inputs, settings and parameters
(`/data/plan_runs.jsonl`, also `GET /api/plan/runs?hours=N`). This feeds one of them to the
planner again, checks that it decides what it decided then, and lets you change a setting to see
what the plan would have been.

Usage:
    python scripts/replay_plan.py runs.jsonl --at 2026-10-31T16:00        # closest run at/after
    python scripts/replay_plan.py runs.json --at 2026-10-31T16:00 --set comfort_cost_sek_per_c_hour=2
    python scripts/replay_plan.py runs.jsonl --list

`runs` is the jsonl file or the JSON array from the API. `--set` replaces a PlanSettings field.
"""

import argparse
import json
import sys
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.theria.plan.plan_run import PlanRun, replay, run_from_dict  # noqa: E402
from core.theria.plan.settings import PlanSettings  # noqa: E402


def load(path: str) -> list[dict]:
    text = Path(path).read_text()
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def moment(text: str) -> datetime:
    at = datetime.fromisoformat(text)
    return at if at.tzinfo else at.replace(tzinfo=UTC)


def overridden(settings: PlanSettings, assignments: list[str]) -> PlanSettings:
    types = {f.name: f.type for f in fields(PlanSettings)}
    changes = {}
    for item in assignments:
        key, _, value = item.partition("=")
        if key not in types:
            sys.exit(f"unknown setting {key!r}")
        kind = types[key]
        changes[key] = (
            value.lower() == "true" if kind is bool else (kind)(value)  # type: ignore[operator]
        )
    return replace(settings, **changes)


def table(run: PlanRun, result, label: str) -> None:
    print(f"\n{label}: objective {result.objective_sek:.2f} kr")
    print("hour(UTC)        price  rts  closed  kWh   room_end  binding")
    for h in result.hours:
        print(
            f"{h.start:%m-%d %H:%M}  {h.price_sek_per_kwh:7.2f}  {h.rts_c:4.1f}  "
            f"{'yes' if h.valves_closed else '   '}   {h.energy_kwh:5.2f}  "
            f"{h.room_c_end:7.2f}  {h.reason.get('binding')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs")
    parser.add_argument("--at", help="ISO time (UTC unless it has an offset)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()

    bodies = load(args.runs)
    if args.list or not args.at:
        for b in bodies:
            print(b["computed_at"], b["mode"], f"{b['decided'][0]['rts_c']:.1f}")
        return
    target = moment(args.at)
    chosen = next(
        (b for b in bodies if datetime.fromisoformat(b["computed_at"]) >= target), None
    )
    if chosen is None:
        sys.exit("no run at or after that time")
    run = run_from_dict(chosen)
    print(f"run computed {run.computed_at.isoformat()} ({run.mode})")

    again = replay(run)
    now = [h.rts_c for h in again.hours]
    then = [d["rts_c"] for d in run.decided]
    print("replay matches the stored decision:", now == then)
    table(run, again, "replay")
    if args.set:
        changed = replay(run, overridden(run.settings, args.set))
        table(run, changed, "with " + " ".join(args.set))


if __name__ == "__main__":
    main()
