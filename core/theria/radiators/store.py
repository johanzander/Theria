"""The radiator preferences that Johan edits in the UI, saved as one JSON file.

The add-on options give the defaults; a change made in the UI is saved next to them and survives
restarts. A damaged or no longer valid file is ignored (the defaults stand), as `SettingsManager` does.
"""

import json
import logging
import os
from typing import Any

from core.theria.exceptions import ConfigurationError
from core.theria.radiators.settings import RadiatorPrefs

logger = logging.getLogger(__name__)


class PrefsStore:
    def __init__(self, defaults: RadiatorPrefs, path: str | None):
        self._defaults = defaults
        self._path = path
        self._saved: dict[str, Any] = {}
        self.current = defaults
        self._load()

    def _load(self) -> None:
        if not self._path or not os.path.exists(self._path):
            return
        try:
            with open(self._path) as f:
                saved = json.load(f)
            self.current = self._merged(saved)
            self._saved = saved
        except (OSError, ValueError, ConfigurationError):
            logger.warning("ignoring the saved radiator preferences %s", self._path)

    def _merged(self, values: dict[str, Any]) -> RadiatorPrefs:
        return RadiatorPrefs.from_dict({**self._defaults.to_dict(), **values})

    def is_default(self) -> bool:
        return not self._saved

    def update(self, values: dict[str, Any]) -> RadiatorPrefs:
        """Check and save a change; raises ConfigurationError and changes nothing when invalid."""
        merged = {**self._saved, **values}
        candidate = self._merged(merged)
        self._saved = merged
        self.current = candidate
        self._save()
        return candidate

    def reset(self) -> RadiatorPrefs:
        self._saved = {}
        self.current = self._defaults
        if self._path and os.path.exists(self._path):
            os.remove(self._path)
        return self.current

    def _save(self) -> None:
        if not self._path:
            return
        temporary = f"{self._path}.tmp"
        with open(temporary, "w") as f:
            json.dump(self._saved, f, indent=1)
        os.replace(temporary, self._path)
