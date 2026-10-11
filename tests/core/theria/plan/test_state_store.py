from dataclasses import replace
from datetime import UTC, datetime

import pytest

from core.theria.exceptions import StateStoreError
from core.theria.plan.calibration import FittedModels, PumpFit, RoomFit
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS
from core.theria.plan.room_model import PRIOR_ROOM_PARAMS
from core.theria.plan.state_store import PlanState, PlanStateStore

from .fits import daily_fit

NOW = datetime(2026, 10, 12, 3, 0, tzinfo=UTC)


def fitted() -> FittedModels:
    return FittedModels(
        pump=PumpFit(PRIOR_PUMP_PARAMS, 400, 0.5, 5000, 0.8),
        room=RoomFit(PRIOR_ROOM_PARAMS, daily_fit()),
        room_error=None,
        fitted_at=NOW,
        window_start=datetime(2023, 10, 12, tzinfo=UTC),
        window_end=NOW,
    )


def test_missing_file_is_an_empty_state(tmp_path):
    assert PlanStateStore(str(tmp_path / "state.json")).load() == PlanState(None, None)


def test_a_rejected_room_fit_is_stored_as_an_error_without_a_room_model(tmp_path):
    path = str(tmp_path / "state.json")
    models = replace(fitted(), room=None, room_error="ModelFitError: no variation")
    PlanStateStore(path).save(PlanState(mode="shadow", models=models))
    loaded = PlanStateStore(path).load().models
    assert loaded.room is None and loaded.room_error == "ModelFitError: no variation"
    assert loaded.pump == fitted().pump


def test_mode_and_models_survive_a_restart(tmp_path):
    path = str(tmp_path / "state.json")
    PlanStateStore(path).save(PlanState(mode="off", models=fitted()))
    loaded = PlanStateStore(path).load()
    assert loaded.mode == "off"
    assert loaded.models == fitted()


@pytest.mark.parametrize(
    "text",
    [
        "{broken",
        '{"mode": "bogus", "models": null}',
        '{"mode": "off", "models": {"pump": {}}}',
    ],
)
def test_damaged_files_raise_instead_of_being_ignored(tmp_path, text):
    path = tmp_path / "state.json"
    path.write_text(text)
    with pytest.raises(StateStoreError):
        PlanStateStore(str(path)).load()


def test_unwritable_location_raises(tmp_path):
    store = PlanStateStore(str(tmp_path / "missing_dir" / "state.json"))
    with pytest.raises(StateStoreError):
        store.save(PlanState(mode="shadow", models=None))


def test_the_night_cooling_model_survives_a_restart(tmp_path):
    from datetime import date

    from core.theria.plan.night_drift import NightDrift, NightObservation

    drift = NightDrift(
        -0.5,
        0.06,
        -0.2,
        19.5,
        6,
        120,
        0.12,
        (NightObservation(date(2026, 10, 10), 3.0, -0.3, -0.32),),
    )
    store = PlanStateStore(str(tmp_path / "state.json"))
    store.save(PlanState(mode="shadow", models=replace(fitted(), night_drift=drift)))
    loaded = store.load().models
    assert loaded.night_drift == drift and loaded.night_drift_error is None


def test_the_warm_up_fit_survives_a_restart(tmp_path):
    from core.theria.plan.warmup import WarmupFit

    warmup = WarmupFit(1.2, 0.08, 1.2, 0.17, 0.14, 0.2, 1.1, 0.1, 380, 90)
    store = PlanStateStore(str(tmp_path / "state.json"))
    store.save(PlanState(mode="shadow", models=replace(fitted(), warmup=warmup)))
    loaded = store.load().models
    assert loaded.warmup == warmup and loaded.warmup_error is None
