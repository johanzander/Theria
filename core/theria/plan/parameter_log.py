"""A persistent history of what the plan has learned: one line per change.

Every refit of the models and every change of the night margin or the stage appends a record
with all the adjustable parameters at that moment, so that their movement over time can be
read afterwards (the state file only keeps the latest values). The file is JSON lines.
"""

import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class ParameterLog:
    def __init__(self, path: str):
        self._path = path

    def append(
        self, at: datetime, kind: str, note: str, params: dict[str, Any]
    ) -> None:
        record = {"at": at.isoformat(), "kind": kind, "note": note, "params": params}
        with open(self._path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Records oldest first (the latest `limit` of them); damaged lines are skipped."""
        try:
            with open(self._path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        records: list[dict[str, Any]] = []
        for number, line in enumerate(lines, start=1):
            try:
                record = json.loads(line)
                record["at"] = datetime.fromisoformat(record["at"])
                records.append(record)
            except (ValueError, KeyError, TypeError):
                logger.warning("skipping damaged line %d in %s", number, self._path)
        return records if limit is None else records[-limit:]
