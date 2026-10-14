"""The plan as it was decided, hour by hour, kept so that the charts can show plan and outcome together.

The service recomputes the plan every cycle, so the plan that was applied for an hour is the first one
computed in it. One JSON line per hour; later plans for the same hour are not kept.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanRecord:
    hour: datetime
    computed_at: datetime
    rts_c: float
    room_c: float  # expected at the end of the hour: the level plus the sun's swing
    energy_kwh: float
    baseline_rts_c: float
    baseline_room_c: float
    baseline_energy_kwh: float
    price_sek_per_kwh: float
    # added later: older lines in the file lack them
    flow_c: float | None = None
    baseline_flow_c: float | None = None
    binding: str | None = None


def _to_dict(record: PlanRecord) -> dict[str, Any]:
    body = asdict(record)
    body["hour"] = record.hour.isoformat()
    body["computed_at"] = record.computed_at.isoformat()
    return body


def _from_dict(body: dict[str, Any]) -> PlanRecord:
    return PlanRecord(
        **{
            **body,
            "hour": datetime.fromisoformat(body["hour"]),
            "computed_at": datetime.fromisoformat(body["computed_at"]),
        }
    )


class PlanRecordStore:
    def __init__(self, path: str):
        self._path = path
        self._records: dict[datetime, PlanRecord] = {}
        try:
            with open(path) as f:
                for line in f:
                    try:
                        record = _from_dict(json.loads(line))
                    except (ValueError, KeyError, TypeError):
                        logger.warning("skipping damaged line in %s", path)
                        continue
                    self._records.setdefault(record.hour, record)
        except FileNotFoundError:
            pass

    def append(self, record: PlanRecord) -> None:
        """Keeps the first record of an hour; later ones are ignored."""
        if record.hour in self._records:
            return
        self._records[record.hour] = record
        with open(self._path, "a") as f:
            f.write(json.dumps(_to_dict(record)) + "\n")

    def read(self, start: datetime, stop: datetime) -> dict[datetime, PlanRecord]:
        """The records of the hours in [start, stop)."""
        return {h: r for h, r in self._records.items() if start <= h < stop}
