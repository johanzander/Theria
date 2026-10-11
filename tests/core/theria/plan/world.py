"""A synthetic house with known pump and room parameters, served as history."""

from datetime import UTC, datetime, timedelta

import numpy as np

from core.theria.plan.pump_model import (
    PumpParams,
    aux_kwh_per_hour,
    energy_kwh_per_hour,
    flow_target,
)
from core.theria.plan.room_model import RoomParams, RoomState, simulate
from core.theria.plan.weather import WeatherHour

END = datetime(2026, 3, 1, 0, tzinfo=UTC)

TRUE_PUMP = PumpParams(
    flow_intercept=41.0,
    flow_slope=-0.7,
    rts_ref=18.0,
    flow_per_rts=1.9,
    energy_intercept=0.4,
    energy_per_outdoor=-0.15,
    energy_per_rts=0.12,
    aux_onset_c=44.0,
    aux_per_flow_c=0.15,
)
TRUE_ROOM = RoomParams(  # gain 0.4 C per C of flow, closes 20 % of the gap per day
    tau_slab_h=1.0,
    k_slab=0.4 * 0.2 / 24,
    k_out=0.6 * 0.2 / 24,
    k_sun=0.0,
    sun_lag_h=1.0,
    drift_c_per_h=0.0,
)
LK_SETPOINT = 19.5
SONOFF_OFFSET_C = 1.7  # the second-opinion sensor reads warmer than the LK sensors
ENTITIES = {
    "outdoor": "sensor.outdoor",
    "room": "sensor.lk_bibliotek_temperature",
    "sonoff": "sensor.sonoff_a480045bf5_temperature",
    "lk": "climate.lk_bibliotek_thermostat",
    "rts": "climate.room_temp_setpoint",
    "setpoint": "sensor.heating_setpoint",
    "valve": "binary_sensor.switch_valve_1",
    "flow": "sensor.radiator_forward",
    "counter": "sensor.compr_cons_heating",
    "aux_counter": "sensor.aux_cons_heating",
}


class World:
    """Hourly truth for `days` days ending at END, plus the sources that serve it."""

    def __init__(
        self,
        days: int = 150,
        seed: int = 3,
        hot_water_hours: bool = False,
        hot_water_blocks: bool = False,
        room_params: RoomParams = TRUE_ROOM,
        late_room_params: RoomParams | None = None,
        late_days: int = 0,
    ):
        rng = np.random.default_rng(seed)
        n = days * 24
        self.start = END - timedelta(hours=n)
        hrs = np.arange(n)
        self.t_out = (
            3
            + 5 * np.sin(2 * np.pi * hrs / 24 - 1)
            + 3 * np.sin(2 * np.pi * hrs / (24 * 9))
        )
        self.sun = np.clip(300 * np.sin(2 * np.pi * (hrs % 24 - 6) / 24), 0, None) * (
            0.4 + 0.6 * (np.sin(hrs / 41) > 0)
        )
        steps = rng.uniform(15.5, 21.5, size=n // 60 + 1)
        self.rts = np.repeat(steps, 60)[:n].round(1)
        self.setpoint = flow_target(TRUE_PUMP, self.t_out, self.rts)
        self.hot_water = np.zeros(n)
        if hot_water_hours:
            self.hot_water[(hrs % 24) == 4] = 1.0
        self.energy = energy_kwh_per_hour(TRUE_PUMP, self.t_out, self.rts)
        self.aux = aux_kwh_per_hour(TRUE_PUMP, self.t_out, self.rts)
        self.valve_events: list[tuple[datetime, float]] | None = None
        if hot_water_blocks:
            # every day 04:05-04:40 the pump makes hot water: the heating counter only sees
            # the remaining 25 minutes' worth, and the hour's mean flow is polluted
            blocked = (hrs % 24) == 4
            self.energy = np.where(blocked, self.energy * 0.3, self.energy)
            self.valve_events = [(self.start - timedelta(hours=1), 0.0)]
            for day in range(n // 24):
                base = self.start + timedelta(days=day, hours=4)
                self.valve_events += [
                    (base + timedelta(minutes=5), 1.0),
                    (base + timedelta(minutes=40), 0.0),
                ]
        self.counter = 1000.0 + np.cumsum(np.round(self.energy * 4) / 4)
        self.aux_counter = 50.0 + np.cumsum(np.round(self.aux * 4) / 4)
        self.flow = self.setpoint - 1.0
        split = n - late_days * 24 if late_room_params is not None else n
        first = simulate(
            room_params,
            RoomState(19.0, 24.0, 0.0),
            self.flow[:split],
            self.t_out[:split],
            self.sun[:split],
            LK_SETPOINT,
        )
        parts = [first]
        if split < n:
            # continue from the state at the end of the first part (slab ~ room)
            parts.append(
                simulate(
                    late_room_params,
                    RoomState(float(first[-1]), float(first[-1]), 0.0),
                    self.flow[split:],
                    self.t_out[split:],
                    self.sun[split:],
                    LK_SETPOINT,
                )
            )
        end_of_hour = np.concatenate(parts)
        # the history is labelled by hour: hour i holds the temperature at the START of hour i
        self.room = np.concatenate([[19.0], end_of_hour[:-1]])

    def points(self, values: np.ndarray) -> list[tuple[datetime, float]]:
        return [
            (self.start + timedelta(hours=i, minutes=30), float(v))
            for i, v in enumerate(values)
        ]


class FakeSource:
    def __init__(
        self,
        world: World,
        overrides: dict[tuple[str, str | None], np.ndarray] | None = None,
    ):
        self._world = world
        w = world
        self._series = {
            (ENTITIES["outdoor"], None): w.t_out,
            (ENTITIES["room"], None): w.room,
            (ENTITIES["sonoff"], None): w.room + SONOFF_OFFSET_C,
            (ENTITIES["lk"], "temperature"): np.full(len(w.t_out), LK_SETPOINT),
            (ENTITIES["rts"], "temperature"): w.rts,
            (ENTITIES["setpoint"], None): w.setpoint,
            (ENTITIES["valve"], None): w.hot_water,
            (ENTITIES["flow"], None): w.flow,
            (ENTITIES["counter"], None): w.counter,
            (ENTITIES["aux_counter"], None): w.aux_counter,
        }
        self._series.update(overrides or {})

    def series(self, entity_id, start, stop, field=None):
        if entity_id == ENTITIES["valve"] and self._world.valve_events is not None:
            return self._world.valve_events
        return self._world.points(self._series[(entity_id, field)])


class FakeWeatherHistory:
    def __init__(self, world: World):
        self._world = world

    def history(self, start, stop):
        w = self._world
        return [
            WeatherHour(
                w.start + timedelta(hours=i), float(w.sun[i]), 50.0, float(w.t_out[i])
            )
            for i in range(len(w.sun))
        ]
