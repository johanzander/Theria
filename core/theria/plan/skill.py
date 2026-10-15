"""How well the model predicts, in the hours without sun, decides how much freedom the plan gets.

Every cycle the disturbance estimator compares the room with what the model said from the measured
inputs. One 15-minute comparison is mostly sensor noise (the readings come in tenths of a degree), so
the evidence is the *systematic* error: the mean per six-hour block, and the mean of the absolute block
means over the last three days. A model that is right on average earns the full swing of the setpoint
around its base; one that is not is held to half of it. With no evidence yet the plan has the half.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from core.theria.plan.disturbance import Innovation

BLOCK = timedelta(hours=6)
WINDOW = timedelta(hours=72)
MIN_PER_BLOCK = 8  # 15-minute samples (2 hours) before a block counts
MIN_BLOCKS = 6  # a day and a half of evidence
GOOD_C_PER_H = 0.02  # systematic error at or below this: full authority
BAD_C_PER_H = 0.08  # at or above this: the minimum
MIN_AUTHORITY = 0.5


class SkillTracker:
    def __init__(self) -> None:
        self._samples: dict[int, list[float]] = defaultdict(list)  # block number -> rates

    @staticmethod
    def _block(moment: datetime) -> int:
        return int(moment.timestamp() // BLOCK.total_seconds())

    def add(self, innovation: Innovation) -> None:
        if not innovation.dark:
            return
        self._samples[self._block(innovation.at)].append(innovation.rate_c_per_h)

    def error(self, now: datetime) -> float | None:
        oldest = self._block(now - WINDOW)
        for block in [b for b in self._samples if b < oldest]:
            del self._samples[block]
        means = [
            abs(float(np.mean(values)))
            for values in self._samples.values()
            if len(values) >= MIN_PER_BLOCK
        ]
        return float(np.mean(means)) if len(means) >= MIN_BLOCKS else None

    def authority(self, now: datetime, minimum: float = MIN_AUTHORITY) -> float:
        """The share of the setpoint range the plan may use: `minimum` with no evidence or a poor
        model, up to 1 when the model predicts well."""
        error = self.error(now)
        if error is None:
            return minimum
        share = (BAD_C_PER_H - error) / (BAD_C_PER_H - GOOD_C_PER_H)
        return minimum + (1.0 - minimum) * float(np.clip(share, 0.0, 1.0))

    def export_state(self) -> dict[str, Any]:
        return {str(block): values for block, values in self._samples.items()}

    def load_state(self, state: dict[str, Any]) -> None:
        self._samples = defaultdict(
            list, {int(block): [float(v) for v in values] for block, values in state.items()}
        )
