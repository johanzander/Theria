import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.calibration import Calibrator, fit_pump
from core.theria.plan.calibration_data import load_calibration_inputs
from core.theria.plan.pump_model import energy_kwh_per_hour
from core.theria.plan.room_model import RoomParams

from .world import (
    END,
    ENTITIES,
    SONOFF_OFFSET_C,
    TRUE_PUMP,
    FakeSource,
    FakeWeatherHistory,
    World,
)


def calibrator(settings, world, overrides=None, days=150):
    return Calibrator(
        FakeSource(world, overrides), FakeWeatherHistory(world), settings, days=days
    )


def test_inputs_are_aligned_hour_by_hour_and_hot_water_hours_are_marked(settings):
    world = World(days=20, hot_water_hours=True)
    inputs = load_calibration_inputs(
        FakeSource(world), FakeWeatherHistory(world), settings, END, days=20
    )
    assert len(inputs.hours) == 20 * 24
    assert inputs.hours[0] == world.start and inputs.hours[-1].hour == 23
    assert np.allclose(inputs.t_out_c, world.t_out, atol=1e-9)
    assert np.allclose(inputs.sun_wm2, world.sun)
    # the valve is sampled mid-hour here, so an hour of hot water shows up as a share
    assert inputs.hot_water[4] > 0.0 and inputs.hot_water[8] == 0.0
    # energy comes from counter differences: hour 0 has no previous counter value
    assert np.isnan(inputs.energy_kwh[0])
    assert np.allclose(inputs.energy_kwh[1:], np.round(world.energy[1:] * 4) / 4)


def test_hot_water_hours_do_not_break_the_fit(settings):
    world = World(hot_water_hours=True)
    fitted = calibrator(settings, world).run(END)
    assert fitted.pump.params.flow_slope == pytest.approx(
        TRUE_PUMP.flow_slope, abs=0.02
    )


def test_a_constant_room_setpoint_cannot_identify_the_energy_model(settings):
    world = World()
    constant = np.full(len(world.rts), 18.0)
    cal = calibrator(settings, world, {(ENTITIES["rts"], "temperature"): constant})
    with pytest.raises(ModelFitError):
        cal.run(END)


def test_too_little_heating_history_is_rejected(settings):
    world = World(days=20)
    with pytest.raises(ModelFitError):
        Calibrator(FakeSource(world), FakeWeatherHistory(world), settings, days=20).run(
            END
        )


def test_hours_with_a_short_hot_water_block_are_not_fitted_as_heating(settings):
    # Hot water runs 04:05-04:40 every day; the valve is off again at the end of the hour,
    # so the last state of that hour says "heating" although the hour was mostly not.
    world = World(hot_water_blocks=True)
    inputs = load_calibration_inputs(
        FakeSource(world), FakeWeatherHistory(world), settings, END, days=150
    )
    assert inputs.hot_water[4] == pytest.approx(35 / 60)
    assert inputs.hot_water[5] == 0.0
    pump = fit_pump(inputs).params
    # intercept and slope trade off; what matters is the predicted consumption
    for t_out, rts in ((0.0, 17.0), (3.0, 18.5), (6.0, 20.0)):
        assert float(energy_kwh_per_hour(pump, t_out, rts)) == pytest.approx(
            float(energy_kwh_per_hour(TRUE_PUMP, t_out, rts)), abs=0.03
        )


def test_the_pump_is_recovered_from_the_history(settings):
    pump = calibrator(settings, World()).run(END).pump.params
    assert pump.flow_slope == pytest.approx(TRUE_PUMP.flow_slope, abs=0.01)
    assert pump.flow_intercept + pump.flow_per_rts * (
        18.0 - pump.rts_ref
    ) == pytest.approx(TRUE_PUMP.flow_intercept, abs=0.05)
    assert pump.energy_per_rts == pytest.approx(TRUE_PUMP.energy_per_rts, abs=0.01)


def three_seasons(**kwargs):
    return World(days=3 * 365, **kwargs)


def test_the_rooms_response_to_flow_is_recovered_from_several_winters(settings):
    fitted = calibrator(settings, three_seasons(), days=3 * 365).run(END)
    response = fitted.room.response
    assert response.gain_c_per_c == pytest.approx(0.4, abs=0.06)
    assert response.rate_per_day == pytest.approx(0.2, abs=0.04)
    assert len(response.seasons) >= 2
    assert fitted.room.params.k_slab == pytest.approx(
        0.4 * response.rate_per_day / 24, rel=0.2
    )


def test_a_room_that_does_not_follow_the_flow_is_rejected_but_the_pump_fit_is_kept(
    settings,
):
    deaf = RoomParams(
        1.0, 0.0, 0.05 / 24, 0.0, 1.0, 0.0
    )  # loses heat, ignores the floor
    fitted = calibrator(settings, three_seasons(room_params=deaf), days=3 * 365).run(
        END
    )
    assert fitted.room is None and "response" in fitted.room_error
    assert fitted.pump.params.flow_slope == pytest.approx(
        TRUE_PUMP.flow_slope, abs=0.02
    )


def test_seasons_that_disagree_about_the_response_are_rejected(settings):
    weak = RoomParams(1.0, 0.08 * 0.2 / 24, 0.92 * 0.2 / 24, 0.0, 1.0, 0.0)
    world = three_seasons(
        late_room_params=weak, late_days=243
    )  # the 2025/26 season from 1 July
    fitted = calibrator(settings, world, days=3 * 365).run(END)
    assert fitted.room is None and "differ" in fitted.room_error


def test_a_short_history_gives_no_room_model(settings):
    fitted = calibrator(settings, World()).run(END)  # 150 days: one season
    assert fitted.room is None and "seasons" in fitted.room_error


def test_the_room_series_is_extended_back_with_the_second_opinion_sensor(settings):
    world = three_seasons()
    lk_only_recently = world.room.copy()
    lk_only_recently[: -120 * 24] = (
        np.nan
    )  # LK sensors exist only for the last 120 days
    source = FakeSource(world, {(ENTITIES["room"], None): lk_only_recently})
    inputs = load_calibration_inputs(
        source, FakeWeatherHistory(world), settings, END, days=3 * 365
    )
    early = slice(200 * 24, 300 * 24)
    assert np.isnan(inputs.room_c[early]).all()  # the LK sensors have nothing here
    assert np.allclose(inputs.room_reference_c[early], world.room[early], atol=0.05)
    assert np.allclose(
        inputs.room_reference_c[-100 * 24 :], world.room[-100 * 24 :], atol=0.05
    )
    assert SONOFF_OFFSET_C > 0


def test_the_older_sensor_is_scaled_to_the_amplitude_of_the_lk_sensors_before_it_is_used(
    settings,
):
    world = three_seasons()
    lk_only_recently = world.room.copy()
    lk_only_recently[: -120 * 24] = np.nan
    mean = float(np.mean(world.room))
    wiggly = (
        mean + 1.25 * (world.room - mean) + SONOFF_OFFSET_C
    )  # swings 25 % more than LK
    source = FakeSource(
        world,
        {
            (ENTITIES["room"], None): lk_only_recently,
            (ENTITIES["sonoff"], None): wiggly,
        },
    )
    inputs = load_calibration_inputs(
        source, FakeWeatherHistory(world), settings, END, days=3 * 365
    )
    early = slice(200 * 24, 300 * 24)
    assert np.allclose(inputs.room_reference_c[early], world.room[early], atol=0.15)


def test_the_electric_heater_is_fitted_from_its_own_counter(settings):
    fitted = calibrator(settings, World()).run(END)
    assert fitted.pump.params.aux_per_flow_c == pytest.approx(0.15, abs=0.03)
    assert fitted.pump.aux_rows > 20


def test_aux_energy_is_aligned_like_the_compressor_energy(settings):
    world = World(days=20)
    inputs = load_calibration_inputs(
        FakeSource(world), FakeWeatherHistory(world), settings, END, days=20
    )
    assert np.isnan(inputs.aux_kwh[0])
    assert np.allclose(inputs.aux_kwh[1:], np.round(world.aux[1:] * 4) / 4)


def test_a_house_where_the_heater_never_runs_gets_no_heater_term(settings):
    world = World()
    quiet = np.full(len(world.aux_counter), 50.0)
    fitted = calibrator(settings, world, {(ENTITIES["aux_counter"], None): quiet}).run(
        END
    )
    assert fitted.pump.params.aux_per_flow_c == 0.0
