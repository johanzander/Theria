"""Persisting the plan mode and fitted models across restarts."""

import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from core.theria.exceptions import StateStoreError
from core.theria.plan.calibration import FittedModels, PumpFit, RoomFit
from core.theria.plan.daily_room import DailyRoomFit, SeasonFit
from core.theria.plan.pump_model import PumpParams
from core.theria.plan.room_model import RoomParams
from core.theria.plan.settings import SUPPORTED_MODES
from core.theria.plan.stages import StageState


@dataclass(frozen=True)
class PlanState:
    mode: str | None
    models: FittedModels | None
    stage: StageState | None = None
    stage_checked_on: date | None = None
    executor: dict[str, Any] | None = None  # what the executors remember
    lk: dict[str, Any] | None = None


def _stage_to_dict(s: StageState) -> dict[str, Any]:
    return {
        "index": s.index,
        "stage_started_on": s.stage_started_on.isoformat(),
        "revert_until": None if s.revert_until is None else s.revert_until.isoformat(),
    }


def _stage_from_dict(d: dict[str, Any]) -> StageState:
    revert = d["revert_until"]
    return StageState(
        index=int(d["index"]),
        stage_started_on=date.fromisoformat(d["stage_started_on"]),
        revert_until=None if revert is None else date.fromisoformat(revert),
    )


def _response_to_dict(r: DailyRoomFit) -> dict[str, Any]:
    return {
        "gain_c_per_c": r.gain_c_per_c,
        "rate_per_day": r.rate_per_day,
        "time_constant_days": r.time_constant_days,
        "seasons": [s.__dict__ for s in r.seasons],
        "windows": r.windows,
        "cv_rmse_c": {str(h): v for h, v in r.cv_rmse_c.items()},
        "cv_persistence_rmse_c": {
            str(h): v for h, v in r.cv_persistence_rmse_c.items()
        },
    }


def _response_from_dict(d: dict[str, Any]) -> DailyRoomFit:
    return DailyRoomFit(
        gain_c_per_c=float(d["gain_c_per_c"]),
        rate_per_day=float(d["rate_per_day"]),
        time_constant_days=float(d["time_constant_days"]),
        seasons=tuple(SeasonFit(**s) for s in d["seasons"]),
        windows=int(d["windows"]),
        cv_rmse_c={int(h): float(v) for h, v in d["cv_rmse_c"].items()},
        cv_persistence_rmse_c={
            int(h): float(v) for h, v in d["cv_persistence_rmse_c"].items()
        },
    )


def _models_to_dict(m: FittedModels) -> dict[str, Any]:
    return {
        "pump": {
            "params": m.pump.params.__dict__,
            "flow_rows": m.pump.flow_rows,
            "flow_rmse_c": m.pump.flow_rmse_c,
            "energy_rows": m.pump.energy_rows,
            "energy_rmse_kwh": m.pump.energy_rmse_kwh,
            "aux_rows": m.pump.aux_rows,
        },
        "room": (
            None
            if m.room is None
            else {
                "params": m.room.params.__dict__,
                "response": _response_to_dict(m.room.response),
            }
        ),
        "room_error": m.room_error,
        "fitted_at": m.fitted_at.isoformat(),
        "window_start": m.window_start.isoformat(),
        "window_end": m.window_end.isoformat(),
    }


def _models_from_dict(d: dict[str, Any]) -> FittedModels:
    pump, room = d["pump"], d["room"]
    return FittedModels(
        pump=PumpFit(
            params=PumpParams(**pump["params"]),
            flow_rows=int(pump["flow_rows"]),
            flow_rmse_c=float(pump["flow_rmse_c"]),
            energy_rows=int(pump["energy_rows"]),
            energy_rmse_kwh=float(pump["energy_rmse_kwh"]),
            aux_rows=int(pump.get("aux_rows", 0)),  # absent in older state files
        ),
        room=(
            None
            if room is None
            else RoomFit(
                params=RoomParams(**room["params"]),
                response=_response_from_dict(room["response"]),
            )
        ),
        room_error=d["room_error"],
        fitted_at=datetime.fromisoformat(d["fitted_at"]),
        window_start=datetime.fromisoformat(d["window_start"]),
        window_end=datetime.fromisoformat(d["window_end"]),
    )


class PlanStateStore:
    """A small JSON file. A missing file is an empty state; a damaged file is an error."""

    def __init__(self, path: str):
        self._path = path

    def load(self) -> PlanState:
        if not os.path.exists(self._path):
            return PlanState(mode=None, models=None)
        try:
            with open(self._path) as f:
                raw = json.load(f)
            mode = raw.get("mode")
            if mode is not None and mode not in SUPPORTED_MODES:
                raise StateStoreError(f"unsupported mode {mode!r} in {self._path}")
            models = raw.get("models")
            stage, checked = raw.get("stage"), raw.get("stage_checked_on")
            return PlanState(
                mode=mode,
                models=None if models is None else _models_from_dict(models),
                stage=None if stage is None else _stage_from_dict(stage),
                stage_checked_on=(
                    None if checked is None else date.fromisoformat(checked)
                ),
                executor=raw.get("executor"),
                lk=raw.get("lk"),
            )
        except (
            OSError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            AttributeError,
        ) as e:
            raise StateStoreError(f"cannot read plan state {self._path}: {e}") from e

    def save(self, state: PlanState) -> None:
        payload = {
            "mode": state.mode,
            "models": None if state.models is None else _models_to_dict(state.models),
            "stage": None if state.stage is None else _stage_to_dict(state.stage),
            "executor": state.executor,
            "lk": state.lk,
            "stage_checked_on": (
                None
                if state.stage_checked_on is None
                else state.stage_checked_on.isoformat()
            ),
        }
        temporary = f"{self._path}.tmp"
        try:
            with open(temporary, "w") as f:
                json.dump(payload, f, indent=1)
            os.replace(temporary, self._path)
        except OSError as e:
            raise StateStoreError(f"cannot write plan state {self._path}: {e}") from e
