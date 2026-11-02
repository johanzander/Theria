"""The salon's temperature: the plain mean of all valid points.

The floor and ceiling sensors differ from the thermostats on purpose, so no point is corrected to a
reference as the plan's room temperature does.
"""

from collections.abc import Collection
from dataclasses import dataclass

from core.theria.sensor_guard import SensorReport

MIN_POINTS = 2


@dataclass(frozen=True)
class SalonReading:
    mean_c: float | None
    used: tuple[str, ...]
    skipped: dict[str, str]  # entity -> why it was left out
    thermostat_mean_c: float | None  # the mean of the thermostat sensors alone
    unknown: bool  # fewer than two valid points: the zone is not cut


def read_salon(
    reports: list[SensorReport], thermostat_sensors: Collection[str]
) -> SalonReading:
    values: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for r in reports:
        if r.usable and r.value_c is not None:
            values[r.temperature_entity] = r.value_c
        else:
            reason = "; ".join(i.message for i in r.issues) or r.status
            skipped[r.temperature_entity] = reason
    thermostat_values = [v for e, v in values.items() if e in thermostat_sensors]
    unknown = len(values) < MIN_POINTS
    return SalonReading(
        mean_c=None if unknown else sum(values.values()) / len(values),
        used=tuple(values),
        skipped=skipped,
        thermostat_mean_c=(
            sum(thermostat_values) / len(thermostat_values)
            if thermostat_values
            else None
        ),
        unknown=unknown,
    )
