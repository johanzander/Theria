"""Every plan computation with everything it was computed from, so that a plan can be replayed.

The plan is a pure function of its inputs, the settings it ran on and the model parameters. A run
keeps all of them next to what the plan decided; `replay` feeds them to the planner again, so that
a strange hour ("why pre-heat at the dearest price?") can be reproduced, taken apart and tried with
another setting. One JSON line per run, kept for a few days.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from core.theria.plan.planner import Plan, PlanInputs, plan
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.room_model import RoomParams, RoomState
from core.theria.plan.settings import PlanSettings

logger = logging.getLogger(__name__)

KEEP = timedelta(days=7)


@dataclass(frozen=True)
class PlanRun:
    computed_at: datetime
    mode: str
    inputs: PlanInputs
    settings: PlanSettings
    pump: PumpParams
    room: RoomParams
    # what the plan decided then, per hour: start, rts_c, valves_closed, energy_kwh, binding
    decided: list[dict[str, Any]]
    objective_sek: float


def _settings_to_dict(settings: PlanSettings) -> dict[str, Any]:
    body = asdict(settings)
    body["room_actuators"] = [
        {"room": room, "entities": list(entities)}
        for room, entities in settings.room_actuators.items()
    ]
    return body


def _array(values: Any) -> list[float] | None:
    return None if values is None else [float(v) for v in values]


def run_to_dict(run: PlanRun) -> dict[str, Any]:
    i = run.inputs
    return {
        "computed_at": run.computed_at.isoformat(),
        "mode": run.mode,
        "inputs": {
            "hour_starts": [h.isoformat() for h in i.hour_starts],
            "price_sek_per_kwh": _array(i.price_sek_per_kwh),
            "extreme": [bool(e) for e in i.extreme],
            "t_out_c": _array(i.t_out_c),
            "sun_wm2": _array(i.sun_wm2),
            "initial": asdict(i.initial),
            "previous_rts_c": i.previous_rts_c,
            "extra_drift_c_per_h": _array(i.extra_drift_c_per_h),
        },
        "settings": _settings_to_dict(run.settings),
        "pump": asdict(run.pump),
        "room": asdict(run.room),
        "decided": run.decided,
        "objective_sek": run.objective_sek,
    }


def run_from_dict(body: dict[str, Any]) -> PlanRun:
    i = body["inputs"]
    drift = i["extra_drift_c_per_h"]
    return PlanRun(
        computed_at=datetime.fromisoformat(body["computed_at"]),
        mode=body["mode"],
        inputs=PlanInputs(
            hour_starts=[datetime.fromisoformat(h) for h in i["hour_starts"]],
            price_sek_per_kwh=np.array(i["price_sek_per_kwh"]),
            extreme=np.array(i["extreme"], dtype=bool),
            t_out_c=np.array(i["t_out_c"]),
            sun_wm2=np.array(i["sun_wm2"]),
            initial=RoomState(**i["initial"]),
            previous_rts_c=i["previous_rts_c"],
            extra_drift_c_per_h=None if drift is None else np.array(drift),
        ),
        settings=PlanSettings.from_dict(body["settings"]),
        pump=PumpParams(**body["pump"]),
        room=RoomParams(**body["room"]),
        decided=body["decided"],
        objective_sek=body["objective_sek"],
    )


def make_run(
    computed_at: datetime,
    mode: str,
    inputs: PlanInputs,
    settings: PlanSettings,
    pump: PumpParams,
    room: RoomParams,
    result: Plan,
) -> PlanRun:
    return PlanRun(
        computed_at=computed_at,
        mode=mode,
        inputs=inputs,
        settings=settings,
        pump=pump,
        room=room,
        decided=[
            {
                "start": h.start.isoformat(),
                "rts_c": h.rts_c,
                "valves_closed": h.valves_closed,
                "energy_kwh": h.energy_kwh,
                "binding": h.reason.get("binding"),
            }
            for h in result.hours
        ],
        objective_sek=result.objective_sek,
    )


def replay(run: PlanRun, settings: PlanSettings | None = None) -> Plan:
    """The plan again from what the run kept; `settings` replaces the ones it ran on."""
    return plan(run.inputs, settings or run.settings, run.pump, run.room)


class PlanRunStore:
    def __init__(self, path: str, now: datetime):
        self._path = path
        self._lines = self._load(now)

    def _load(self, now: datetime) -> list[tuple[datetime, str]]:
        kept: list[tuple[datetime, str]] = []
        try:
            with open(self._path) as f:
                for line in f:
                    try:
                        at = datetime.fromisoformat(json.loads(line)["computed_at"])
                    except (ValueError, KeyError, TypeError):
                        logger.warning("skipping damaged line in %s", self._path)
                        continue
                    if now - at <= KEEP:
                        kept.append((at, line.rstrip("\n")))
        except FileNotFoundError:
            return kept
        with open(self._path, "w") as f:
            f.writelines(line + "\n" for _, line in kept)
        return kept

    def append(self, run: PlanRun) -> None:
        line = json.dumps(run_to_dict(run))
        self._lines.append((run.computed_at, line))
        with open(self._path, "a") as f:
            f.write(line + "\n")

    def read(self, since: datetime) -> list[dict[str, Any]]:
        return [json.loads(line) for at, line in self._lines if at >= since]
