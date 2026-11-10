from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from core.theria.exceptions import ModelFitError
from core.theria.plan.energy_forecast import (
    EnergyForecaster,
    EnergyForecastStore,
    EnergyHistory,
    forecast_energy,
    load_energy_history,
    summarize_days,
    trailing_means,
)
from core.theria.plan.series import interpolated_counter_energy
from core.theria.plan.weather import WeatherHour

TZ = "Europe/Stockholm"
START = datetime(2026, 1, 1, 0, tzinfo=UTC)


def world(days=120, heat=lambda t, r: np.maximum(0.25 * (16 - t), 0) + 0.2 * (r - 18)):
    n = days * 24
    hours = [START + timedelta(hours=i) for i in range(n)]
    rng = np.random.default_rng(1)
    t_out = 4 + 6 * np.sin(np.arange(n) / (24 * 9)) + rng.normal(0, 0.5, n)
    setpoint = 18 + np.round(np.sin(np.arange(n) / 50))
    return EnergyHistory(
        hours=hours,
        t_out_c=t_out,
        sun_wm2=np.zeros(n),
        wind_kmh=np.full(n, 10.0),
        setpoint_c=setpoint,
        flow_c=40 - 0.7 * t_out + 3 * (setpoint - 18),
        heating_kwh=heat(t_out, setpoint),
        hot_water_kwh=np.full(n, 0.4),
    )


def day_range(h, day):
    tz = ZoneInfo(TZ)
    return np.array(
        [i for i, x in enumerate(h.hours) if x.astimezone(tz).date() == day]
    )


def cut_for(h, day):
    tz = ZoneInfo(TZ)
    before = day - timedelta(days=1)
    return next(
        i
        for i, x in enumerate(h.hours)
        if x.astimezone(tz).date() == before and x.astimezone(tz).hour == 14
    )


DAY = date(2026, 4, 20)


def run(h, day=DAY):
    return forecast_energy(h, cut_for(h, day), day_range(h, day), TZ)


def test_the_day_is_predicted_from_outdoor_temperature_and_setpoint():
    h = world()
    target = day_range(h, DAY)
    truth = h.heating_kwh[target] + h.hot_water_kwh[target]
    assert run(h)["simple"].sum() == pytest.approx(truth.sum(), rel=0.03)


def test_a_colder_day_costs_more():
    h = world()
    target = day_range(h, DAY)
    mild = run(h)["simple"].sum()
    cold = world()
    cold.t_out_c[target] -= 5
    assert run(cold)["simple"].sum() > mild + 5


def test_a_higher_setpoint_costs_more():
    h = world()
    target = day_range(h, DAY)
    base = run(h)["simple"].sum()
    h.setpoint_c[target] += 2
    assert run(h)["simple"].sum() > base


def test_nothing_measured_after_the_cut_is_used():
    h = world()
    day = DAY
    before = run(h, day)
    cut = cut_for(h, day)
    h.heating_kwh[cut:] = 99.0
    h.hot_water_kwh[cut:] = 99.0
    h.flow_c[cut:] = 99.0
    after = run(h, day)
    for name in before:
        assert after[name] == pytest.approx(before[name])


def test_hot_water_is_the_mean_per_clock_hour_with_weekends_apart():
    h = world()
    tz = ZoneInfo(TZ)
    local = [x.astimezone(tz) for x in h.hours]
    h.heating_kwh[:] = 0.0
    h.hot_water_kwh[:] = [
        (2.0 if x.hour == 7 else 0.0)
        if x.weekday() < 5
        else (2.0 if x.hour == 10 else 0.0)
        for x in local
    ]
    h.t_out_c[:] = 20.0
    monday = date(2026, 4, 20)
    saturday = date(2026, 4, 25)
    weekday = run(h, monday)["simple"]
    weekend = run(h, saturday)["simple"]
    assert weekday[7] == pytest.approx(2.0, abs=0.05)
    assert weekday[10] == pytest.approx(0.0, abs=0.05)
    assert weekend[10] == pytest.approx(2.0, abs=0.05)


def test_recent_consumption_above_the_model_lifts_the_forecast_by_half_the_ratio():
    plain = world()
    lifted = world()
    cut = cut_for(lifted, DAY)
    lifted.heating_kwh[cut - 24 * 3 : cut] *= 1.44
    target = day_range(plain, DAY)
    ratio = run(lifted)["simple"].sum() / run(plain)["simple"].sum()
    heating_share = plain.heating_kwh[target].sum() / (
        plain.heating_kwh[target].sum() + plain.hot_water_kwh[target].sum()
    )
    assert ratio == pytest.approx(1 + 0.2 * heating_share, abs=0.05)


def test_a_day_with_a_clock_change_has_its_real_number_of_hours():
    day = date(2026, 3, 29)  # 23 hours in Sweden
    assert len(run(world(days=120), day)["simple"]) == 23


def test_a_gap_in_the_outdoor_history_does_not_spoil_later_hours():
    h = world()
    h.t_out_c[500:510] = np.nan
    assert np.isfinite(run(h)["simple"]).all()


def test_the_reference_forecasts_are_the_48_hour_mean_and_the_7_day_profile():
    h = world()
    cut = cut_for(h, DAY)
    last48 = (h.heating_kwh + h.hot_water_kwh)[cut - 48 : cut]
    out = run(h)
    assert out["mean_48h"] == pytest.approx(np.full(24, last48.mean()))
    assert len(out["profile_7d"]) == 24


def test_too_little_history_is_an_error():
    h = world(days=120)
    with pytest.raises(ModelFitError):
        forecast_energy(h, 24 * 3, np.arange(24 * 4, 24 * 5), TZ)


# The counter


def points(*rows):
    return [(START + timedelta(hours=h), v) for h, v in rows]


def test_a_counter_step_is_spread_over_the_hours_between_readings():
    energy = interpolated_counter_energy(
        points((0, 100), (2, 101), (4, 102)), START, 4, 8.0, 8
    )
    assert energy == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_counter_garbage_is_left_out():
    energy = interpolated_counter_energy(
        points((0, 100), (1, 101), (1.5, 3), (2, 102), (3, 103)), START, 3, 8.0, 8
    )
    assert energy == pytest.approx([1.0, 1.0, 1.0])


def test_a_counter_reset_adds_no_energy():
    readings = [(h, 500 + h) for h in range(0, 400)] + [
        (h, h - 400) for h in range(400, 800)
    ]
    energy = interpolated_counter_energy(points(*readings), START, 799, 8.0, 8)
    assert np.nanmax(energy) <= 1.0 + 1e-9


def test_hours_in_a_long_gap_are_unknown():
    energy = interpolated_counter_energy(
        points((0, 100), (1, 101), (20, 110), (21, 111)), START, 21, 8.0, 8
    )
    assert energy[0] == pytest.approx(1.0)
    assert np.isnan(energy[10])


# Scoring


def line(day, weather, simple, actual):
    return {
        "day": day,
        "weather": weather,
        "forecasts": {"simple": simple},
        "actual": actual,
    }


def test_the_summary_gives_day_error_and_bias_per_method_and_weather():
    rows = summarize_days(
        [
            line("2026-11-01", "forecast", [2.0, 2.0], [1.0, 1.0]),
            line("2026-11-02", "forecast", [1.0, 1.0], [2.0, 2.0]),
            line("2026-11-03", "measured", [3.0, 3.0], [2.0, 2.0]),
        ]
    )
    assert rows == [
        {
            "method": "simple",
            "weather": "forecast",
            "days": 2,
            "day_error_kwh": 2.0,
            "bias_kwh": 0.0,
            "share": pytest.approx(4 / 6),
        },
        {
            "method": "simple",
            "weather": "measured",
            "days": 1,
            "day_error_kwh": 2.0,
            "bias_kwh": 2.0,
            "share": pytest.approx(0.5),
        },
    ]


def test_a_day_with_an_unmeasured_hour_is_not_scored():
    assert (
        summarize_days([line("2026-11-01", "forecast", [1.0, 1.0], [1.0, None])]) == []
    )


# Issuing


def forecast_weather(first, hours, temperature):
    return [
        WeatherHour(first + timedelta(hours=i), 0.0, 0.0, temperature, 10.0)
        for i in range(hours)
    ]


class Harness:
    def __init__(self, tmp_path, now):
        self.now = now
        self.loads = 0
        self.store = EnergyForecastStore(str(tmp_path / "energy.jsonl"))
        self.forecaster = EnergyForecaster(
            self.load,
            self.weather,
            self.store,
            TZ,
            lambda: self.now,
            lambda hour: None,
            backfill_days=2,
        )

    def load(self, start, stop):
        self.loads += 1
        n = int((stop - start) / timedelta(hours=1))
        hours = [start + timedelta(hours=i) for i in range(n)]
        known = np.array([x < self.now for x in hours])
        absolute = np.array([x.timestamp() / 3600 for x in hours])
        t = np.where(known, 5.0 + 3 * np.sin(absolute / 100.0), np.nan)

        def measured(values):
            return np.where(known, values, np.nan)

        return EnergyHistory(
            hours=hours,
            t_out_c=t,
            sun_wm2=measured(np.zeros(n)),
            wind_kmh=measured(np.full(n, 10.0)),
            setpoint_c=measured(np.full(n, 18.0)),
            flow_c=measured(40 - 0.7 * np.nan_to_num(t)),
            heating_kwh=measured(0.25 * (16 - np.nan_to_num(t))),
            hot_water_kwh=measured(np.full(n, 0.4)),
        )

    def weather(self):
        first = self.now.replace(minute=0, second=0, microsecond=0) - timedelta(
            hours=24
        )
        return forecast_weather(first, 24 * 4, 0.0)


def test_tomorrow_is_issued_once_after_14_with_the_weather_forecast(tmp_path):
    h = Harness(tmp_path, datetime(2026, 4, 20, 13, 30, tzinfo=UTC))  # 15:30 local
    assert h.forecaster.run_due() == 3
    lines = {x["day"]: x for x in h.store.read()}
    assert lines["2026-04-21"]["weather"] == "forecast"
    assert lines["2026-04-19"]["weather"] == "measured"
    assert lines["2026-04-18"]["weather"] == "measured"
    assert len(lines["2026-04-21"]["forecasts"]["simple"]) == 24
    loads = h.loads
    assert h.forecaster.run_due() == 0
    assert h.loads == loads


def test_tomorrow_is_not_issued_before_14(tmp_path):
    h = Harness(tmp_path, datetime(2026, 4, 20, 10, 0, tzinfo=UTC))  # 12:00 local
    h.forecaster.run_due()
    assert "2026-04-21" not in {x["day"] for x in h.store.read()}


def test_tomorrows_weather_is_shifted_to_what_the_outdoor_sensor_shows(tmp_path):
    now = datetime(2026, 4, 20, 13, 30, tzinfo=UTC)
    h = Harness(tmp_path, now)
    h.forecaster.run_due()
    lines = {x["day"]: x for x in h.store.read()}
    history = h.load(now - timedelta(days=1), now)
    sensor = float(np.nanmean(history.t_out_c))
    # the forecast says 0 C, so tomorrow stands where the sensor has stood
    assert sum(lines["2026-04-21"]["forecasts"]["simple"]) == pytest.approx(
        (0.25 * (16 - sensor) + 0.4) * 24, rel=0.15
    )


def test_a_planned_setpoint_is_used_when_there_is_one(tmp_path):
    h = Harness(tmp_path, datetime(2026, 4, 20, 13, 30, tzinfo=UTC))
    seen = []

    def planned(hour):
        seen.append(hour)
        return 19.0

    h.forecaster._planned_setpoint = planned
    h.forecaster.run_due()
    tomorrow = next(x for x in h.store.read() if x["day"] == "2026-04-21")
    assert tomorrow and seen


def test_the_view_joins_forecasts_with_what_was_measured(tmp_path):
    h = Harness(tmp_path, datetime(2026, 4, 20, 13, 30, tzinfo=UTC))
    h.forecaster.run_due()
    view = h.forecaster.view(7)
    days = {d["day"]: d for d in view["days"]}
    assert days["2026-04-19"]["actual"][0] == pytest.approx(
        0.25 * (16 - 5) + 0.4, abs=1.5
    )
    assert all(v is None for v in days["2026-04-21"]["actual"])
    assert days["2026-04-21"]["clock"] == list(range(24))
    assert view["summary"][0]["days"] == 2


# Loading


class OneSensorSource:
    def __init__(self):
        self.asked = None

    def series(self, entity_id, start, stop, field=None):
        return []

    def hourly_means(self, entity_id, start, stop, plausible=None):
        self.asked = plausible
        return [(start + timedelta(hours=i), 5.0) for i in range(48)]


def test_outdoor_garbage_is_left_out_before_the_hours_are_averaged(settings):
    source = OneSensorSource()
    load_energy_history(
        source, settings, lambda a, b: [], START, START + timedelta(days=2)
    )
    assert source.asked == (-35.0, 45.0)


# The model the plan uses


def test_the_heating_model_gives_the_forecasts_heating_part():
    from core.theria.plan.energy_forecast import fit_heating_model

    h = world()
    cut, target = cut_for(h, DAY), day_range(h, DAY)
    model = fit_heating_model(h, cut, TZ, DAY)
    t24, t72 = trailing_means(h.t_out_c)
    table = model.table(
        h.t_out_c[target],
        t24[target],
        t72[target],
        h.sun_wm2[target],
        h.wind_kmh[target],
        np.array([h.setpoint_c[target[0]]]),
    )
    hot = run(h)["simple"] - table[:, 0]
    assert hot == pytest.approx(np.full(len(target), 0.4), abs=0.02)


def test_the_model_costs_more_at_a_higher_setpoint():
    from core.theria.plan.energy_forecast import fit_heating_model

    h = world()
    model = fit_heating_model(h, cut_for(h, DAY), TZ, DAY)
    table = model.table(
        np.array([5.0]),
        np.array([5.0]),
        np.array([5.0]),
        np.zeros(1),
        np.full(1, 10.0),
        np.array([17.0, 18.0, 19.0]),
    )
    assert table[0, 0] < table[0, 1] < table[0, 2]


def test_the_model_survives_a_restart():
    from core.theria.plan.energy_forecast import HeatingModel, fit_heating_model

    h = world()
    model = fit_heating_model(h, cut_for(h, DAY), TZ, DAY)
    assert HeatingModel.from_dict(model.to_dict()) == model


def test_the_forecaster_fits_a_model_for_the_plan_even_when_nothing_is_due(tmp_path):
    h = Harness(tmp_path, datetime(2026, 4, 20, 10, 0, tzinfo=UTC))
    h.forecaster._backfill_days = 0
    assert h.forecaster.heating_model() is None
    h.forecaster.run_due()
    assert h.forecaster.heating_model() is not None
    loads = h.loads
    h.forecaster.run_due()
    assert h.loads == loads  # fitted once a day
