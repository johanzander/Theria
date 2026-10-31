"""A JSON-lines file that keeps the last few days: one object per line, stamped with a time."""

import json
import logging
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class RecentLines:
    def __init__(self, path: str, now: datetime, keep: timedelta, stamp: str):
        self._path = path
        self._stamp = stamp
        self._lines = self._load(now, keep)

    def _load(self, now: datetime, keep: timedelta) -> list[tuple[datetime, str]]:
        kept: list[tuple[datetime, str]] = []
        try:
            with open(self._path) as f:
                for line in f:
                    try:
                        at = datetime.fromisoformat(json.loads(line)[self._stamp])
                    except (ValueError, KeyError, TypeError):
                        logger.warning("skipping damaged line in %s", self._path)
                        continue
                    if now - at <= keep:
                        kept.append((at, line.rstrip("\n")))
        except FileNotFoundError:
            return kept
        with open(self._path, "w") as f:
            f.writelines(line + "\n" for _, line in kept)
        return kept

    def append(self, body: dict[str, Any]) -> None:
        line = json.dumps(body)
        self._lines.append((datetime.fromisoformat(body[self._stamp]), line))
        with open(self._path, "a") as f:
            f.write(line + "\n")

    def read(self, since: datetime) -> list[dict[str, Any]]:
        return [json.loads(line) for at, line in self._lines if at >= since]
