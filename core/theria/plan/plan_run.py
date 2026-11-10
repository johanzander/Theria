"""Every plan computation with everything it was computed from, so that a plan can be replayed.

The plan is a pure function of its inputs, the settings it ran on and the model parameters. A run
keeps all of them next to what the plan decided; `replay` feeds them to the heat shift again, so that
a strange hour ("why pre-heat at the dearest price?") can be reproduced, taken apart and tried with
another setting. One JSON line per run, kept for a few days.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from core.theria.plan.heat_shift import plan_shift
from core.theria.plan.plan_model import Plan, PlanInputs
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.recent_log import RecentLines
from core.theria.plan.room_model import RoomParams, RoomState
from core.theria.plan.settings import PlanSettings
from core.theria.plan.shift_rules import ShiftReadings

KEEP = timedelta(days=7)


@dataclass(frozen=True)
class PlanRun:
    computed_at: datetime
    mode: str
    inputs: PlanInputs
    settings: PlanSettings
    pump: PumpParams
    room: RoomParams
    # what the plan decided then, per hour: start, rts_c, valves_closed, energy_kwh, room_c, flow_c,
    # binding and the options it weighed
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


def _optional_array(values: Any) -> np.ndarray | None:
    return None if values is None else np.array(values, dtype=float)


def run_to_dict(run: PlanRun) -> dict[str, Any]:
    i = run.inputs
    return {
        "computed_at": run.computed_at.isoformat(),
        "mode": run.mode,
        "inputs": {
            "hour_starts": [h.isoformat() for h in i.hour_starts],
            "price_sek_per_kwh": _array(i.price_sek_per_kwh),
            "t_out_c": _array(i.t_out_c),
            "sun_wm2": _array(i.sun_wm2),
            "initial": asdict(i.initial),
            "previous_rts_c": i.previous_rts_c,
            "extra_drift_c_per_h": _array(i.extra_drift_c_per_h),
            "past_prices_sek_per_kwh": _array(i.past_prices_sek_per_kwh),
            "shift": None if i.shift is None else i.shift.to_dict(),
            "energy_setpoints_c": _array(i.energy_setpoints_c),
            "energy_kwh_table": (
                None
                if i.energy_kwh_table is None
                else [_array(row) for row in i.energy_kwh_table]
            ),
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
    past = i.get("past_prices_sek_per_kwh")  # absent in runs kept before 2026-11-09
    return PlanRun(
        computed_at=datetime.fromisoformat(body["computed_at"]),
        mode=body["mode"],
        inputs=PlanInputs(
            hour_starts=[datetime.fromisoformat(h) for h in i["hour_starts"]],
            price_sek_per_kwh=np.array(i["price_sek_per_kwh"]),
            t_out_c=np.array(i["t_out_c"]),
            sun_wm2=np.array(i["sun_wm2"]),
            initial=RoomState(**i["initial"]),
            previous_rts_c=i["previous_rts_c"],
            extra_drift_c_per_h=None if drift is None else np.array(drift),
            past_prices_sek_per_kwh=None if past is None else np.array(past),
            shift=(
                None if i.get("shift") is None else ShiftReadings.from_dict(i["shift"])
            ),
            # absent in runs kept before 2026-11-10
            energy_setpoints_c=_optional_array(i.get("energy_setpoints_c")),
            energy_kwh_table=_optional_array(i.get("energy_kwh_table")),
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
                "thermostat": h.thermostat,
                "energy_kwh": h.energy_kwh,
                # what the room was expected to read at the end of the hour, and the flow target
                "room_c": h.room_c_end + h.sun_swing_c,
                "flow_c": h.flow_target_c,
                "binding": h.reason.get("binding"),
            }
            for h in result.hours
        ],
        objective_sek=result.objective_sek,
    )


def replay(run: PlanRun, settings: PlanSettings | None = None) -> Plan:
    """The plan again from what the run kept; `settings` replaces the ones it ran on."""
    return plan_shift(run.inputs, settings or run.settings, run.pump, run.room)


class PlanRunStore(RecentLines):
    def __init__(self, path: str, now: datetime):
        super().__init__(path, now, KEEP, "computed_at")

    def append_run(self, run: PlanRun) -> None:
        self.append(run_to_dict(run))
