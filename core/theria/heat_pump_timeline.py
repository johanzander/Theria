"""What the heat pump does: its history from InfluxDB, its current readings from Home Assistant.

Read only. Many H60 sensors are stored when they change, so a value is carried until the next
reading and the last reading before the range starts the series. Readings outside what the sensor
can show are H60 garbage and never enter a bucket. The cumulative energy counters go through the
same filter as the plan's calibration.
"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

from core.theria.ha_client import HAClient
from core.theria.heat_pump_monitor import _read_float, _read_mode
from core.theria.influxdb_helper import query_influx
from core.theria.plan.calibration_data import (
    MAX_AUX_KWH_PER_HOUR,
    MAX_ENERGY_KWH_PER_HOUR,
)
from core.theria.plan.series import clean_counter

MAX_RANGE = timedelta(days=366)
LOCAL_ZONE = ZoneInfo("Europe/Stockholm")

STEPS_S = (60, 300, 900, 1800, 3600, 10800, 21600, 86400)
MAX_POINTS = 2100  # a canvas this wide draws them without losing a peak
PRICE_STEP_S = 900
HOUR_S = 3600
LOOKBACK_S = 90 * 86400  # how far back the value a series starts from may lie
COUNTER_LOOKBACK_S = 14 * 86400  # the counter filter needs readings before the range
PER_DAY_FROM = timedelta(days=14)  # beyond this hourly bars are too thin to read
MAX_SUPPLIED_KWH_PER_HOUR = 26.0  # AirX 170 at most 17 kW plus the 9 kW electric heater
GARBAGE_MARGIN = (
    1.5  # one reading carries a gap's energy, so a bucket may exceed the limit
)

Rows = list[list[Any]]


@dataclass(frozen=True)
class Series:
    key: str
    measurement: str
    aggregate: str  # mean: measured; last: a setting that steps; max: on or off
    low: float | None = None
    high: float | None = None
    field: str = "value"

    def _plausible(self) -> str:
        if self.low is None or self.high is None:
            return ""
        return f' AND "{self.field}" >= {self.low} AND "{self.field}" <= {self.high}'

    def bucket_query(self, start_s: int, stop_s: int, step_s: int) -> str:
        return (
            f'SELECT {self.aggregate}("{self.field}") FROM "{self.measurement}" '
            f"WHERE time >= {start_s}s AND time < {stop_s}s{self._plausible()} "
            f"GROUP BY time({step_s}s) fill(none)"
        )

    def before_query(self, start_s: int) -> str:
        return (
            f'SELECT "{self.field}" FROM "{self.measurement}" '
            f"WHERE time >= {start_s - LOOKBACK_S}s AND time < {start_s}s{self._plausible()} "
            f"ORDER BY time DESC LIMIT 1"
        )


@dataclass(frozen=True)
class Counter:
    key: str
    measurement: str
    max_kwh_per_hour: float


SERIES = (
    Series("comp", "sensor.compressor_speed_2", "mean", 0, 100),  # % despite the name
    Series("power", "sensor.compressor_speed", "mean", 0, 30),  # kW despite the name
    Series("aux", "sensor.add_heat_status", "mean", 0, 100),
    Series("valve", "binary_sensor.switch_valve_1", "max"),
    Series("outdoor", "sensor.outdoor", "mean", -35, 45),
    Series("flowset", "sensor.heating_setpoint", "last", 15, 65),
    Series("rts", "climate.room_temp_setpoint", "last", 5, 35, "temperature"),
    Series("flow", "sensor.radiator_forward", "mean", 5, 80),
    Series("hc_fwd", "sensor.heat_carrier_forwrd", "mean", 5, 90),
    Series("hc_ret", "sensor.heat_carrier_return", "mean", 5, 90),
    Series("hotgas", "sensor.hot_gas_compr", "mean", -10, 140),
    Series("cond", "sensor.brine_out_condenser", "mean", -30, 90),
    Series("evap", "sensor.brine_in_evaporator", "mean", -30, 60),
    Series("circ", "sensor.circ_pump_speed", "mean", 0, 100),
    Series("dhwset", "sensor.warm_water_setpoint", "last", 20, 80),
    Series("dhwtop", "sensor.warm_water_1_top", "mean", 5, 90),
    Series("dhwmid", "sensor.warm_water_2_mid", "mean", 5, 90),
)
PRICE = Series("price", "sensor.nord_pool_se4_current_price", "mean", -5, 50)
COUNTERS = (
    Counter("supp_heat", "sensor.supp_energy_heating", MAX_SUPPLIED_KWH_PER_HOUR),
    Counter("supp_hw", "sensor.supp_energy_hotwater", MAX_SUPPLIED_KWH_PER_HOUR),
    Counter("cons_heat", "sensor.compr_cons_heating", MAX_ENERGY_KWH_PER_HOUR),
    Counter("cons_hw", "sensor.compr_cons_hotwat", MAX_ENERGY_KWH_PER_HOUR),
    Counter("cons_aux", "sensor.aux_consumption_tot", MAX_AUX_KWH_PER_HOUR),
)


LIVE_KEYS = (
    "outdoor", "evap", "cond", "hotgas", "power", "comp",
    "aux", "flow", "hc_fwd", "hc_ret", "dhwtop", "dhwmid",
)  # fmt: skip
VALVE_ENTITY = "binary_sensor.switch_valve_1"


def _live_value(client: HAClient, series: Series) -> float | None:
    value = _read_float(client, series.measurement)
    if value is None:
        return None
    if series.low is not None and value < series.low:
        return None
    if series.high is not None and value > series.high:
        return None
    return value


def read_pump_now(client: HAClient) -> dict[str, Any]:
    """What the pump's sensors in Home Assistant read now; garbage or unavailable has no value."""
    by_key = {s.key: s for s in SERIES}
    return {
        "values": {key: _live_value(client, by_key[key]) for key in LIVE_KEYS},
        "hot_water": _read_mode(client, VALVE_ENTITY) == "hot_water",
    }


def _step_for(span_s: int) -> int:
    return next(s for s in STEPS_S if span_s / s <= MAX_POINTS)


def _grid(start_s: int, stop_s: int, step_s: int) -> list[int]:
    return list(range(start_s - start_s % step_s, stop_s, step_s))


def _carried(rows: Rows, before: Rows, times_s: list[int]) -> list[float | None]:
    """The value of each bucket: its own, or the last one seen before it."""
    known = {int(t / 1000): v for t, v in rows if v is not None}
    current = before[0][1] if before else None
    out: list[float | None] = []
    for t in times_s:
        current = known.get(t, current)
        out.append(None if current is None else round(float(current), 3))
    return out


def _energy_edges(start: datetime, stop: datetime) -> list[int]:
    """The bucket boundaries of the energy bars: whole hours, or local midnights for long ranges."""
    start_s, stop_s = int(start.timestamp()), int(stop.timestamp())
    if stop - start <= PER_DAY_FROM:
        last = -(-stop_s // HOUR_S) * HOUR_S
        return list(range(start_s - start_s % HOUR_S, last + HOUR_S, HOUR_S))
    day = start.astimezone(LOCAL_ZONE).date()
    edges: list[int] = []
    while not edges or edges[-1] < stop_s:
        midnight = datetime(day.year, day.month, day.day, tzinfo=LOCAL_ZONE)
        edges.append(int(midnight.timestamp()))
        day += timedelta(days=1)
    return edges


def _rise_per_bucket(
    rows: Rows, counter: Counter, edges: list[int]
) -> list[float | None]:
    """What the counter rose by between each pair of edges; None when that cannot be real."""
    # the highest reading of an hour is the counter at the end of that hour
    points = [
        (datetime.fromtimestamp(t / 1000 + HOUR_S, UTC), float(v))
        for t, v in rows
        if v is not None
    ]
    good = clean_counter(points, counter.max_kwh_per_hour)
    stamps = [p[0].timestamp() for p in good]

    def reading_at(edge: int) -> float | None:
        i = bisect_right(stamps, edge)
        return good[i - 1][1] if i else None

    out: list[float | None] = []
    for left, right in pairwise(edges):
        a, b = reading_at(left), reading_at(right)
        limit = counter.max_kwh_per_hour * (right - left) / HOUR_S * GARBAGE_MARGIN
        out.append(None if a is None or b is None or not 0 <= b - a <= limit else b - a)
    return out


def build_timeline(start: datetime, stop: datetime) -> dict[str, Any]:
    """All series for [start, stop) at a resolution that suits the length of the range."""
    start_s, stop_s = int(start.timestamp()), int(stop.timestamp())
    step_s = _step_for(stop_s - start_s)
    price_step_s = max(step_s, PRICE_STEP_S)
    edges = _energy_edges(start, stop)

    statements: list[str] = []
    for s in SERIES:
        statements += [
            s.bucket_query(start_s - start_s % step_s, stop_s, step_s),
            s.before_query(start_s - start_s % step_s),
        ]
    price_from = start_s - start_s % price_step_s
    statements += [
        PRICE.bucket_query(price_from, stop_s, price_step_s),
        PRICE.before_query(price_from),
    ]
    statements += [
        f'SELECT max("value") FROM "{c.measurement}" '
        f"WHERE time >= {edges[0] - COUNTER_LOOKBACK_S}s AND time < {edges[-1]}s "
        f"GROUP BY time({HOUR_S}s) fill(none)"
        for c in COUNTERS
    ]
    answers = iter(query_influx(statements))

    times_s = _grid(start_s, stop_s, step_s)
    series = {s.key: _carried(next(answers), next(answers), times_s) for s in SERIES}
    series["hc_delta"] = [
        None if a is None or b is None else round(a - b, 3)
        for a, b in zip(series["hc_fwd"], series["hc_ret"], strict=True)
    ]
    price_times_s = _grid(start_s, stop_s, price_step_s)
    prices = _carried(next(answers), next(answers), price_times_s)
    energy = {c.key: _rise_per_bucket(next(answers), c, edges) for c in COUNTERS}

    return {
        "start_ms": start_s * 1000,
        "stop_ms": stop_s * 1000,
        "step_ms": step_s * 1000,
        "times": [t * 1000 for t in times_s],
        "series": series,
        "energy": {
            "times": [
                t * 1000 for t in edges
            ],  # one more than values: the last edge closes the last bar
            "series": energy,
        },
        "prices": {"times": [t * 1000 for t in price_times_s], "values": prices},
    }
