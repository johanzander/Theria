"""Aligning Home Assistant history to a regular hourly UTC grid and cleaning it."""

from bisect import bisect_right
from datetime import UTC, datetime, timedelta
from itertools import pairwise

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
Point = tuple[datetime, float]


def hour_floor(moment: datetime) -> datetime:
    """The start of the UTC hour containing `moment` (must be timezone-aware)."""
    return moment.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def hourly_grid(start: datetime, n_hours: int) -> list[datetime]:
    first = hour_floor(start)
    return [first + timedelta(hours=i) for i in range(n_hours)]


def to_hourly(
    points: list[Point],
    start: datetime,
    n_hours: int,
    how: str = "mean",
    ffill_hours: int = 0,
) -> FloatArray:
    """Resample irregular points to hourly values; NaN where there is no data.

    how: "mean" or "last" within the hour. `ffill_hours` carries the last value forward
    over that many empty hours (for sensors that only report on change).
    """
    if how not in ("mean", "last"):
        raise ValueError(f"how must be 'mean' or 'last', got {how!r}")
    first = hour_floor(start)
    sums = np.zeros(n_hours)
    counts = np.zeros(n_hours)
    last = np.full(n_hours, np.nan)
    for moment, value in sorted(points, key=lambda p: p[0]):
        index = int((moment.astimezone(UTC) - first) / timedelta(hours=1))
        if 0 <= index < n_hours and np.isfinite(value):
            sums[index] += value
            counts[index] += 1
            last[index] = value
    out = np.full(n_hours, np.nan)
    filled = counts > 0
    out[filled] = sums[filled] / counts[filled] if how == "mean" else last[filled]
    if ffill_hours > 0:
        gap = 0
        for i in range(1, n_hours):
            if np.isnan(out[i]) and np.isfinite(out[i - 1]) and gap < ffill_hours:
                out[i] = out[i - 1]
                gap += 1
            elif np.isfinite(out[i]):
                gap = 0
    return out


def interpolate_gaps(values: FloatArray, max_gap_hours: int) -> FloatArray:
    """Linear interpolation over NaN runs of at most `max_gap_hours` between finite values."""
    out = np.array(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(out))
    for left, right in pairwise(finite):
        gap = right - left - 1
        if 0 < gap <= max_gap_hours:
            out[left + 1 : right] = np.linspace(out[left], out[right], gap + 2)[1:-1]
    return out


def despike(values: FloatArray, window: int, max_dev: float) -> FloatArray:
    """NaN out values further than `max_dev` from the rolling median of `window` hours."""
    out = np.array(values, dtype=float)
    half = window // 2
    for i in range(len(values)):
        if not np.isfinite(values[i]):
            continue
        neighbourhood = values[max(0, i - half) : i + half + 1]
        neighbourhood = neighbourhood[np.isfinite(neighbourhood)]
        if (
            len(neighbourhood) >= 3
            and abs(values[i] - np.median(neighbourhood)) > max_dev
        ):
            out[i] = np.nan
    return out


def hourly_energy_from_counter(
    counter_kwh: FloatArray, max_kwh_per_hour: float
) -> FloatArray:
    """Hourly consumption from a cumulative counter sampled at the end of each hour.

    Hour i gets counter[i] - counter[i-1]. Counter spikes and resets show up as a
    negative or implausibly large difference and are NaN (the heat pump cannot draw
    more than `max_kwh_per_hour`).
    """
    out = np.full(len(counter_kwh), np.nan)
    diff = np.diff(counter_kwh)
    ok = np.isfinite(diff) & (diff >= 0.0) & (diff <= max_kwh_per_hour)
    out[1:][ok] = diff[ok]
    return out


def step_fraction(
    points: list[Point], start: datetime, n_hours: int, max_carry_hours: int = 48
) -> FloatArray:
    """Share (0-1) of each hour that an on/off sensor was on, from its change events.

    The sensor only reports when it changes, so its state is carried forward from the
    previous event. NaN where the state is unknown: before the first event, or more than
    `max_carry_hours` after the last event that touched the hour.
    """
    first = hour_floor(start)
    events = sorted(
        (moment.astimezone(UTC), 1.0 if value >= 0.5 else 0.0)
        for moment, value in points
        if np.isfinite(value)
    )
    out = np.full(n_hours, np.nan)
    state: float | None = None
    since: datetime | None = None
    next_event = 0
    one_hour = timedelta(hours=1)
    for hour in range(n_hours):
        begin = first + hour * one_hour
        end = begin + one_hour
        while next_event < len(events) and events[next_event][0] <= begin:
            since, state = events[next_event]
            next_event += 1
        edges = [begin]
        values: list[float | None] = [state]
        while next_event < len(events) and events[next_event][0] < end:
            edges.append(events[next_event][0])
            values.append(events[next_event][1])
            since = events[next_event][0]
            next_event += 1
        edges.append(end)
        state = values[-1]  # the state at the end of this hour starts the next
        if any(v is None for v in values) or since is None:
            continue
        if begin - since > timedelta(hours=max_carry_hours):
            continue
        on_seconds = sum(
            (edges[k + 1] - edges[k]).total_seconds() * float(values[k] or 0.0)
            for k in range(len(values))
        )
        out[hour] = on_seconds / one_hour.total_seconds()
    return out


COUNTER_CHUNK = timedelta(days=14)


def _longest_non_decreasing(values: list[float]) -> list[int]:
    """Indices of a longest non-decreasing subsequence (patience sorting, O(n log n))."""
    tails: list[float] = []
    tail_index: list[int] = []
    previous = [-1] * len(values)
    for i, value in enumerate(values):
        k = bisect_right(tails, value)
        if k == len(tails):
            tails.append(value)
            tail_index.append(i)
        else:
            tails[k] = value
            tail_index[k] = i
        previous[i] = tail_index[k - 1] if k else -1
    track: list[int] = []
    i = tail_index[-1] if tail_index else -1
    while i != -1:
        track.append(i)
        i = previous[i]
    return track[::-1]


COUNTER_TOLERANCE_KWH = 1.0  # the counters have a resolution of 1 kWh


def _connects(a: Point, b: Point, max_kwh_per_hour: float) -> bool:
    """Whether the counter can have gone from reading `a` to the later reading `b` at a possible consumption."""
    hours = (b[0] - a[0]).total_seconds() / 3600
    return 0.0 <= b[1] - a[1] <= max_kwh_per_hour * hours + COUNTER_TOLERANCE_KWH


def _real_track(chunk: list[Point], max_kwh_per_hour: float) -> list[Point]:
    track = [chunk[i] for i in _longest_non_decreasing([v for _, v in chunk])]
    # the longest run may start or end on garbage that happens to be in order; real readings
    # are joined to their neighbours by a possible consumption
    while len(track) > 1 and not _connects(track[0], track[1], max_kwh_per_hour):
        track.pop(0)
    while len(track) > 1 and not _connects(track[-2], track[-1], max_kwh_per_hour):
        track.pop()
    # and real readings next to the run that lost out to a longer chain of garbage are taken back
    if track:
        before = [p for p in chunk if p[0] < track[0][0]]
        for point in reversed(before):
            if _connects(point, track[0], max_kwh_per_hour):
                track.insert(0, point)
        after = [p for p in chunk if p[0] > track[-1][0]]
        for point in after:
            if _connects(track[-1], point, max_kwh_per_hour):
                track.append(point)
    return track


def clean_counter(points: list[Point], max_kwh_per_hour: float) -> list[Point]:
    """The readings on a cumulative counter's real track; the heat pump's gateway (H60) also returns garbage.

    A cumulative meter only goes up, so the real readings are the longest non-decreasing run. The H60
    now and then answers with a low value (0-16 on counters that really read hundreds), sometimes for
    most of the readings in a stretch, so a median or a step limit would keep the garbage. The run is
    found per `COUNTER_CHUNK` so that a unit change or reset in an old window cannot erase a newer one.
    """
    ordered = sorted((p for p in points if np.isfinite(p[1])), key=lambda p: p[0])
    out: list[Point] = []
    chunk: list[Point] = []
    for point in ordered:
        if chunk and point[0] - chunk[0][0] >= COUNTER_CHUNK:
            out += _real_track(chunk, max_kwh_per_hour)
            chunk = []
        chunk.append(point)
    if chunk:
        out += _real_track(chunk, max_kwh_per_hour)
    return out


def fill_flat_gaps(values: FloatArray) -> FloatArray:
    """Fill NaN runs between two equal readings: a counter that has not moved used nothing.

    Home Assistant only stores a reading when it changes, so a quiet counter leaves long gaps
    that are zero consumption, not missing data. A gap between different readings stays NaN
    because it is not known when in the gap the energy was used.
    """
    out = np.array(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(out))
    for left, right in pairwise(finite):
        if right - left > 1 and out[left] == out[right]:
            out[left + 1 : right] = out[left]
    return out


def hourly_counter_energy(
    points: list[Point],
    start: datetime,
    n_hours: int,
    max_kwh_per_hour: float,
    hold_hours: int,
) -> FloatArray:
    """Hourly consumption from a cumulative counter's readings: cleaned, held over quiet gaps, differenced."""
    last = to_hourly(
        clean_counter(points, max_kwh_per_hour),
        start,
        n_hours,
        how="last",
        ffill_hours=hold_hours,
    )
    return hourly_energy_from_counter(fill_flat_gaps(last), max_kwh_per_hour)
