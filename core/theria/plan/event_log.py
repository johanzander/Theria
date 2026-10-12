"""A persistent copy of the action log: one JSON line per setpoint write, trouble or mode change.

The service keeps the latest entries in memory for the Control tab; this file lets them
survive a restart or an update of the add-on.
"""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class EventLog:
    def __init__(self, path: str):
        self._path = path

    def append(self, event: dict[str, Any]) -> None:
        record = {**event, "at": event["at"].isoformat()}
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self, limit: int) -> list[dict[str, Any]]:
        """The latest `limit` events, oldest first; damaged lines are skipped."""
        try:
            with open(self._path) as f:
                lines = f.readlines()[-limit:]
        except FileNotFoundError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                record = json.loads(line)
                record["at"] = datetime.fromisoformat(record["at"])
                events.append(record)
            except (ValueError, KeyError, TypeError):
                logger.warning("skipping damaged line in %s", self._path)
        return events
