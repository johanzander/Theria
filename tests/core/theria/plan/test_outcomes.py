from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from core.theria.plan.outcomes import OutcomeInputs, day_outcomes, load_outcome_inputs
from core.theria.plan.pump_model import PRIOR_PUMP_PARAMS, energy_kwh_per_hour

START = datetime(2026, 1, 10, 0, tzinfo=UTC)
PUMP = PRIOR_PUMP_PARAMS


def make_inputs(
    rts,
    price,
    t_out=3.0,
    hours=48,
    hot_water=0.0,
    aux=0.0,
    room=19.8,
):
    """Energy is what the pump model says for the actual setpoint (so the model is exact)."""
    rts = np.broadcast_to(np.asarray(rts, dtype=float), (hours,)).copy()
    price = np.broadcast_to(np.asarray(price, dtype=float), (hours,)).copy()
    t = np.full(hours, float(t_out))
    return OutcomeInputs(
        hours=[START + timedelta(hours=i) for i in range(hours)],
        price_sek_per_kwh=price,
        rts_c=rts,
        t_out_c=t,
        compressor_kwh=np.asarray(energy_kwh_per_hour(PUMP, t, rts), dtype=float),
        aux_kwh=np.full(hours, float(aux)),
        hot_water=np.full(hours, float(hot_water)),
        room_c=np.broadcast_to(np.asarray(room, dtype=float), (hours,)).copy(),
    )


def test_no_saving_when_the_setpoint_stays_at_the_base():
    first, second = day_outcomes(make_inputs(18.0, 1.5), PUMP, rts_base_c=18.0)
    assert first.day == date(2026, 1, 10) and second.day == date(2026, 1, 11)
    assert first.saving_sek == pytest.approx(0.0, abs=1e-9)
    assert first.actual_cost_sek == pytest.approx(first.baseline_cost_sek)


def test_shifting_heat_from_expensive_to_cheap_hours_saves_the_price_difference():
    price = np.where(
        np.arange(48) % 24 < 12, 1.0, 3.0
    )  # cheap morning, expensive afternoon
    rts = np.where(np.arange(48) % 24 < 12, 20.0, 16.0)  # more heat in the cheap hours
    outcome = day_outcomes(make_inputs(rts, price), PUMP, rts_base_c=18.0)[0]
    assert outcome.saving_sek > 0
    assert (
        outcome.average_price_paid_sek_per_kwh
        < outcome.average_price_baseline_sek_per_kwh
    )
    # the baseline is the measured energy scaled by the model's change for the setpoint
    expected = sum(
        p * (energy_kwh_per_hour(PUMP, 3.0, 18.0) - energy_kwh_per_hour(PUMP, 3.0, r))
        for p, r in zip(price[:24], rts[:24], strict=True)
    )
    assert outcome.saving_sek == pytest.approx(float(expected), rel=1e-6)


def test_electric_heater_energy_after_a_boost_is_a_cost_with_no_baseline_credit():
    plain = day_outcomes(make_inputs(20.0, 2.0), PUMP, rts_base_c=18.0)[0]
    with_aux = day_outcomes(make_inputs(20.0, 2.0, aux=0.3), PUMP, rts_base_c=18.0)[0]
    assert with_aux.saving_sek == pytest.approx(plain.saving_sek - 24 * 0.3 * 2.0)
    assert with_aux.aux_kwh == pytest.approx(24 * 0.3)


def test_electric_heater_energy_without_a_boost_is_the_same_in_the_baseline():
    plain = day_outcomes(make_inputs(16.0, 2.0), PUMP, rts_base_c=18.0)[0]
    with_aux = day_outcomes(make_inputs(16.0, 2.0, aux=0.3), PUMP, rts_base_c=18.0)[0]
    assert with_aux.saving_sek == pytest.approx(
        plain.saving_sek
    )  # cold-weather heater use


def test_hot_water_hours_are_left_out_because_the_plan_does_not_control_them():
    outcome = day_outcomes(
        make_inputs(16.0, 2.0, hot_water=1.0), PUMP, rts_base_c=18.0
    )[0]
    assert outcome.heating_hours == 0 and outcome.saving_sek == 0.0


def test_hot_water_hours_are_not_missing_data_for_the_coverage():
    inputs = make_inputs(18.0, 1.5, hours=24)
    inputs.hot_water[:12] = 1.0  # half the day the pump made hot water
    day = day_outcomes(inputs, PUMP, rts_base_c=18.0)[0]
    assert day.heating_hours == 12 and day.coverage == pytest.approx(1.0)


def test_days_with_little_data_report_their_coverage_instead_of_a_misleading_total():
    inputs = make_inputs(16.0, 2.0, hours=24)
    price = inputs.price_sek_per_kwh.copy()
    price[:18] = np.nan  # only six hours have a price
    sparse = OutcomeInputs(**{**inputs.__dict__, "price_sek_per_kwh": price})
    outcome = day_outcomes(sparse, PUMP, 18.0)[0]
    assert outcome.heating_hours == 6 and outcome.coverage == pytest.approx(6 / 24)


def test_the_base_setpoint_may_differ_hour_by_hour():
    base = np.where(np.arange(24) < 12, 18.0, 16.0)
    outcome = day_outcomes(make_inputs(base, 1.5, hours=24), PUMP, rts_base_c=base)[0]
    assert outcome.saving_sek == pytest.approx(0.0, abs=1e-9)


class OutcomeSource:
    """History served from small arrays, one point per hour at :30."""

    def __init__(self, series):
        self._series = series

    def series(self, entity_id, start, stop, field=None):
        values = self._series[(entity_id, field)]
        return [
            (START + timedelta(hours=i, minutes=30), float(v))
            for i, v in enumerate(values)
        ]


def test_inputs_are_assembled_from_counters_prices_setpoint_and_valve(settings):
    hours = 6
    counter = np.cumsum([0, 1.5, 2.0, 1.0, 0.5, 2.5]) + 1000.0
    series = {
        (settings.price_entity, None): np.full(hours, 1.5),
        (settings.room_setpoint_entity, "temperature"): np.full(hours, 17.0),
        (settings.outdoor_entity, None): np.full(hours, 3.0),
        (settings.energy_counter_entity, None): counter,
        (settings.aux_energy_entity, None): np.cumsum(np.zeros(hours)) + 40.0,
        (settings.heating_mode_entity, None): np.zeros(hours),
        (settings.room_temp_entities[0], None): np.full(hours, 19.7),
    }
    inputs = load_outcome_inputs(
        OutcomeSource(series), settings, START, START + timedelta(hours=hours)
    )
    assert len(inputs.hours) == hours and inputs.hours[0] == START
    assert np.allclose(inputs.price_sek_per_kwh, 1.5) and np.allclose(
        inputs.rts_c, 17.0
    )
    assert np.isnan(inputs.compressor_kwh[0])  # no counter value before the first hour
    assert np.allclose(inputs.compressor_kwh[1:], np.diff(counter))
    assert np.allclose(inputs.aux_kwh[1:], 0.0)


def test_open_floor_loops_are_counted_per_room(settings):
    from dataclasses import replace

    hours = 4
    settings = replace(
        settings,
        room_actuators={"bib": ("binary_sensor.a", "binary_sensor.b")},
    )
    series = {
        (settings.price_entity, None): np.full(hours, 1.5),
        (settings.room_setpoint_entity, "temperature"): np.full(hours, 17.0),
        (settings.outdoor_entity, None): np.full(hours, 3.0),
        (settings.energy_counter_entity, None): np.arange(hours) + 1000.0,
        (settings.aux_energy_entity, None): np.full(hours, 40.0),
        (settings.heating_mode_entity, None): np.zeros(hours),
        (settings.room_temp_entities[0], None): np.full(hours, 19.7),
        ("binary_sensor.a", None): np.array([1, 1, 0, 0]),
        ("binary_sensor.b", None): np.array([1, 0, 0, 1]),
    }
    inputs = load_outcome_inputs(
        OutcomeSource(series), settings, START, START + timedelta(hours=hours)
    )
    assert inputs.loops_open["bib"][1:].tolist() == [1.5, 0.5, 0.5]  # a loop that switches at :30 counts half


def test_the_rooms_lowest_and_mean_temperature_of_each_day_are_reported():
    room = np.full(48, 19.8)
    room[30] = 19.2  # a dip on the second day
    inputs = make_inputs(18.0, 1.5, room=room)
    first, second = day_outcomes(inputs, PUMP, rts_base_c=18.0)
    assert first.room_min_c == pytest.approx(
        19.8
    ) and second.room_min_c == pytest.approx(19.2)
    assert second.room_mean_c == pytest.approx(np.mean(room[24:]))


def test_heater_use_in_hours_with_a_little_hot_water_still_counts_as_trouble():
    inputs = make_inputs(18.0, 1.5, hours=24)
    inputs.hot_water[5] = 0.3  # mostly heating: not used for the cost, but watched
    inputs.aux_kwh[5] = 2.0
    day = day_outcomes(inputs, PUMP, rts_base_c=18.0)[0]
    assert day.aux_kwh == pytest.approx(2.0)
    assert day.heating_hours == 23  # the hour is still left out of the cost


def test_hours_that_are_mostly_hot_water_are_not_watched_for_heating_trouble():
    inputs = make_inputs(18.0, 1.5, hours=24)
    inputs.hot_water[5] = 0.9
    inputs.aux_kwh[5] = 2.0
    day = day_outcomes(inputs, PUMP, rts_base_c=18.0)[0]
    assert day.aux_kwh == 0.0
