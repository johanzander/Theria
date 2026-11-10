from dataclasses import replace
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.calibration_data import load_calibration_inputs
from core.theria.plan.house_loss import (
    HouseLoss,
    fit_house_loss,
    house_need_kw,
    house_pump_kw,
)
from core.theria.plan.state_store import PlanState, PlanStateStore

from .test_state_store import fitted
from .world import END, FakeSource, FakeWeatherHistory, World

START = datetime(2026, 1, 1, tzinfo=UTC)


def days_of(t_out_by_day, room=19.0, w_per_k=650.0, free_kw=2.5, noise=0.0):
    t_out = np.repeat(np.asarray(t_out_by_day, dtype=float), 24)
    n = len(t_out)
    rng = np.random.default_rng(1)
    heat = w_per_k * (room - t_out) / 1000.0 - free_kw + rng.normal(0, noise, n)
    hours = [START + timedelta(hours=i) for i in range(n)]
    return hours, heat, np.full(n, room), t_out


def test_the_loss_and_the_free_heat_come_back_from_the_heat_counter():
    hours, heat, room, t_out = days_of(np.linspace(-5, 11, 60), noise=0.3)
    house = fit_house_loss(hours, heat, room, t_out)
    assert house.w_per_k == pytest.approx(650.0, rel=0.03)
    assert house.free_kw == pytest.approx(2.5, abs=0.15)
    assert house.days == 60


def test_mild_days_and_days_with_gaps_are_left_out():
    hours, heat, room, t_out = days_of([2.0] * 20 + [14.0] * 30 + [-2.0] * 20)
    heat[: 24 * 5] = np.nan  # five days without the counter
    house = fit_house_loss(hours, heat, room, t_out)
    assert house.days == 35


def test_too_few_cold_days_is_an_error():
    hours, heat, room, t_out = days_of([14.0] * 40 + [3.0] * 10)
    with pytest.raises(ModelFitError):
        fit_house_loss(hours, heat, room, t_out)


def test_an_implausible_loss_is_refused():
    hours, heat, room, t_out = days_of(np.linspace(-5, 11, 60), w_per_k=5000.0)
    with pytest.raises(ModelFitError):
        fit_house_loss(hours, heat, room, t_out)


def test_the_need_is_the_loss_and_the_pump_gives_it_less_the_free_heat():
    house = HouseLoss(w_per_k=600.0, free_kw=2.5, days=100, rmse_kw=1.0)
    assert house_need_kw(house, 20.0, 5.0) == pytest.approx(9.0)
    assert house_pump_kw(house, 20.0, 5.0) == pytest.approx(6.5)
    assert house_pump_kw(house, 20.0, 18.0) == 0.0
    assert house_need_kw(house, float("nan"), 5.0) is None


def test_the_calibration_inputs_carry_the_heat_per_hour(settings):
    world = World(days=40)
    inputs = load_calibration_inputs(
        FakeSource(world), FakeWeatherHistory(world), settings, END, days=40
    )
    known = np.isfinite(inputs.heat_kwh)
    assert known.sum() > 0.9 * len(known)
    assert inputs.heat_kwh[known] == pytest.approx(
        world.heat[-len(known) :][known], abs=1e-6
    )


def test_the_fit_is_kept_across_restarts(tmp_path):
    house = HouseLoss(w_per_k=664.0, free_kw=2.5, days=155, rmse_kw=1.3)
    store = PlanStateStore(str(tmp_path / "state.json"))
    store.save(PlanState(mode="shadow", models=replace(fitted(), house=house)))
    assert store.load().models.house == house
