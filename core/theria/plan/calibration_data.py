"""Hourly calibration data assembled from InfluxDB history and Open-Meteo."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import numpy as np

from core.theria.exceptions import ModelFitError
from core.theria.influxdb_helper import NO_DATA, get_sensor_timeseries
from core.theria.plan.series import (
    FloatArray,
    Point,
    despike,
    hour_floor,
    hourly_counter_energy,
    hourly_grid,
    interpolate_gaps,
    step_fraction,
    to_hourly,
)
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import WeatherHour

MAX_ENERGY_KWH_PER_HOUR = 5.0  # AirX 170 draws about 4.4-5.8 kW at full speed (capacity diagram); anything above is a counter spike
# what the H60 shows for a working pump; it answers 0 or nonsense when a register fails
FLOW_PLAUSIBLE_C = (15.0, 65.0)
OUTDOOR_PLAUSIBLE_C = (-35.0, 45.0)  # the H60 outdoor sensor has read 59.2 C
MAX_AUX_KWH_PER_HOUR = 9.0  # the electric heater (AirBox E9) is at most 9 kW
MAX_HEAT_KWH_PER_HOUR = (
    30.0  # compressor heat plus the heater at most; anything above is a counter spike
)
SETPOINT_FFILL_HOURS = 24 * 60  # setpoints are only logged when they change
SMALL_GAP_HOURS = 3
MIN_OVERLAP_HOURS = 168  # a week of both sensors to know their usual difference
MIN_OVERLAP_DAYS = (
    14  # full days of both sensors to know how much more one of them swings
)


class HistorySource(Protocol):
    def series(
        self, entity_id: str, start: datetime, stop: datetime, field: str | None = None
    ) -> list[Point]:
        """Raw (time, value) points for an entity; `field` picks an attribute."""
        ...

    def hourly_means(
        self,
        entity_id: str,
        start: datetime,
        stop: datetime,
        plausible: tuple[float, float] | None = None,
    ) -> list[Point]:
        """The mean of each hour with data, for sensors too dense to load point by point."""
        ...


class WeatherHistory(Protocol):
    def history(self, start: datetime, stop: datetime) -> list[WeatherHour]: ...


class InfluxHistorySource:
    """HistorySource backed by the existing InfluxDB helper."""

    def series(
        self, entity_id: str, start: datetime, stop: datetime, field: str | None = None
    ) -> list[Point]:
        return self._points(entity_id, start, stop, field_name=field)

    def hourly_means(
        self,
        entity_id: str,
        start: datetime,
        stop: datetime,
        plausible: tuple[float, float] | None = None,
    ) -> list[Point]:
        return self._points(
            entity_id, start, stop, hourly_mean=True, plausible=plausible
        )

    def _points(
        self,
        entity_id: str,
        start: datetime,
        stop: datetime,
        field_name: str | None = None,
        hourly_mean: bool = False,
        plausible: tuple[float, float] | None = None,
    ) -> list[Point]:
        domain, name = entity_id.split(".", 1)
        result = get_sensor_timeseries(
            name,
            start,
            stop,
            domain=domain,
            field_name=field_name,
            hourly_mean=hourly_mean,
            plausible=plausible,
        )
        if result["status"] == "error":
            # A query that worked but found no points is "no data", not a failure.
            if result.get("reason") == NO_DATA:
                return []
            raise ModelFitError(f"InfluxDB query for {entity_id} failed")
        return [(moment, float(value)) for moment, value in result["data"]]


@dataclass(frozen=True)
class CalibrationInputs:
    """Aligned hourly arrays; hour i covers [hours[i], hours[i] + 1 h). NaN = unknown."""

    hours: list[datetime]
    t_out_c: FloatArray
    heating_setpoint_c: FloatArray
    rts_c: FloatArray
    hot_water: FloatArray  # share of the hour the pump made hot water (0 = all heating)
    energy_kwh: FloatArray
    aux_kwh: FloatArray  # electric heater energy per hour
    room_c: FloatArray  # mean of the ground-floor room sensors (LK)
    room_reference_c: FloatArray  # room_c, extended back in time with the older sensors
    flow_c: FloatArray
    lk_setpoint_c: FloatArray
    sun_wm2: FloatArray
    heat_kwh: (
        FloatArray  # heat the pump supplied to the heating per hour (its own counter)
    )

    def tail(self, n_hours: int) -> "CalibrationInputs":
        """The last `n_hours` hours."""
        return CalibrationInputs(
            hours=self.hours[-n_hours:],
            t_out_c=self.t_out_c[-n_hours:],
            heating_setpoint_c=self.heating_setpoint_c[-n_hours:],
            rts_c=self.rts_c[-n_hours:],
            hot_water=self.hot_water[-n_hours:],
            energy_kwh=self.energy_kwh[-n_hours:],
            aux_kwh=self.aux_kwh[-n_hours:],
            room_c=self.room_c[-n_hours:],
            room_reference_c=self.room_reference_c[-n_hours:],
            flow_c=self.flow_c[-n_hours:],
            lk_setpoint_c=self.lk_setpoint_c[-n_hours:],
            sun_wm2=self.sun_wm2[-n_hours:],
            heat_kwh=self.heat_kwh[-n_hours:],
        )


# `Min. utetemperatur` stood at -20 instead of -30: the pump's flow targets were about 2.6 C high
# (8 Nov 12:41 - 9 Nov 17:25 local), so they do not describe its curve
CURVE_CHANGED = (
    datetime(2026, 11, 8, 11, 41, tzinfo=UTC),
    datetime(2026, 11, 9, 16, 25, tzinfo=UTC),
)


def curve_changed(hours: list[datetime]) -> np.ndarray:
    """The hours touched by the wrong curve setting."""
    return np.array(
        [
            CURVE_CHANGED[0] < h + timedelta(hours=1) and h < CURVE_CHANGED[1]
            for h in hours
        ],
        dtype=bool,
    )


def without_curve_change(hours: list[datetime], values: FloatArray) -> FloatArray:
    """The flow targets with the hours of the wrong curve setting left out."""
    return np.where(curve_changed(hours), np.nan, values)


def plausible(points: list[Point], bounds: tuple[float, float]) -> list[Point]:
    """The readings a working sensor can give; the H60 also answers with garbage."""
    return [p for p in points if bounds[0] <= p[1] <= bounds[1]]


def _cleaned(
    points: list[Point], start: datetime, n: int, window: int, max_dev: float
) -> FloatArray:
    hourly = to_hourly(points, start, n, how="mean")
    return interpolate_gaps(despike(hourly, window, max_dev), SMALL_GAP_HOURS)


def _mean_of(columns: list[FloatArray]) -> FloatArray:
    stacked = np.vstack(columns)
    out = np.full(stacked.shape[1], np.nan)
    present = np.isfinite(stacked).any(axis=0)
    out[present] = np.nanmean(stacked[:, present], axis=0)
    return out


def _extended(primary: FloatArray, older: FloatArray) -> FloatArray:
    """`primary`, with gaps filled from `older`, matched to it in level and in swing.

    The older sensor reads warmer (a constant difference) and usually swings more from day
    to day, so it is shifted by the median difference and scaled by the ratio of the day-to-day
    spread of the two over the weeks they overlap. Without a week of overlap (or two weeks of
    full days for the scale) nothing is matched: the level only, or no extension.
    """
    both = np.isfinite(primary) & np.isfinite(older)
    if int(both.sum()) < MIN_OVERLAP_HOURS:
        return primary.copy()
    scale = _swing_ratio(primary, older)
    centre_p = float(np.median(primary[both]))
    centre_o = float(np.median(older[both]))
    return np.where(
        np.isfinite(primary), primary, centre_p + scale * (older - centre_o)
    )


def _swing_ratio(primary: FloatArray, older: FloatArray) -> float:
    """Spread of the primary sensor's daily means over that of the older one (overlap days)."""
    blocks = len(primary) // 24
    p_days, o_days = [], []
    for k in range(blocks):
        p, o = primary[k * 24 : (k + 1) * 24], older[k * 24 : (k + 1) * 24]
        ok = np.isfinite(p) & np.isfinite(o)
        if int(ok.sum()) >= 20:
            p_days.append(float(np.mean(p[ok])))
            o_days.append(float(np.mean(o[ok])))
    if len(p_days) < MIN_OVERLAP_DAYS or np.std(o_days) == 0.0:
        return 1.0
    return float(np.std(p_days) / np.std(o_days))


def load_calibration_inputs(
    source: HistorySource,
    weather: WeatherHistory,
    settings: PlanSettings,
    stop: datetime,
    days: int,
) -> CalibrationInputs:
    """Fetch and align `days` of history ending at `stop` (timezone-aware)."""
    n = days * 24
    end = hour_floor(stop)
    start = end - timedelta(hours=n)
    hours = hourly_grid(start, n)

    def last(entity: str, field: str | None, ffill: int) -> FloatArray:
        return to_hourly(
            source.series(entity, start, end, field),
            start,
            n,
            how="last",
            ffill_hours=ffill,
        )

    t_out = _cleaned(
        source.hourly_means(settings.outdoor_entity, start, end, OUTDOOR_PLAUSIBLE_C),
        start,
        n,
        7,
        6.0,
    )
    rooms = [
        _cleaned(source.hourly_means(e, start, end), start, n, 7, 2.0)
        for e in settings.room_temp_entities
    ]
    lk = [
        last(e, "temperature", SETPOINT_FFILL_HOURS)
        for e in settings.room_thermostat_entities
    ]
    energy_kwh = hourly_counter_energy(
        source.series(settings.energy_counter_entity, start, end),
        start,
        n,
        MAX_ENERGY_KWH_PER_HOUR,
        SMALL_GAP_HOURS,
    )
    aux_kwh = hourly_counter_energy(
        source.series(settings.aux_energy_entity, start, end),
        start,
        n,
        MAX_AUX_KWH_PER_HOUR,
        SMALL_GAP_HOURS,
    )
    heat_kwh = hourly_counter_energy(
        source.series(settings.heat_counter_entity, start, end),
        start,
        n,
        MAX_HEAT_KWH_PER_HOUR,
        SMALL_GAP_HOURS,
    )
    room_c = _mean_of(rooms)
    history = (
        _mean_of(
            [
                _cleaned(source.hourly_means(e, start, end), start, n, 7, 2.0)
                for e in settings.room_history_entities
            ]
        )
        if settings.room_history_entities
        else np.full(n, np.nan)
    )
    sun = np.full(n, np.nan)
    by_start = {
        hour_floor(w.start): w.shortwave_wm2 for w in weather.history(start, end)
    }
    for i, hour in enumerate(hours):
        sun[i] = by_start.get(hour, np.nan)
    return CalibrationInputs(
        hours=hours,
        t_out_c=t_out,
        heating_setpoint_c=without_curve_change(
            hours,
            to_hourly(
                plausible(
                    source.series(settings.flow_setpoint_entity, start, end),
                    FLOW_PLAUSIBLE_C,
                ),
                start,
                n,
                how="last",
                ffill_hours=48,
            ),
        ),
        rts_c=last(settings.room_setpoint_entity, "temperature", SETPOINT_FFILL_HOURS),
        hot_water=step_fraction(
            source.series(settings.heating_mode_entity, start, end), start, n
        ),
        energy_kwh=energy_kwh,
        aux_kwh=aux_kwh,
        room_c=room_c,
        room_reference_c=_extended(room_c, history),
        flow_c=_cleaned(
            source.hourly_means(settings.flow_measured_entity, start, end),
            start,
            n,
            7,
            8.0,
        ),
        lk_setpoint_c=_mean_of(lk),
        sun_wm2=sun,
        heat_kwh=heat_kwh,
    )
