"""The room's usual level, as opposed to a swing the sun has put on it.

The plan keeps the room where it starts (a flat setpoint holds it), so a room that the afternoon
sun has lifted by two or three degrees would be planned as if it stayed there for days, and the
plan would cut the heating for hours that the room will in fact be cold in. The starting point
is therefore the lower of the measured temperature and the median of the last day.
"""

from collections.abc import Sequence

import numpy as np

MIN_HOURS = 12  # hours with a reading needed before the level is trusted


def room_level_c(current_c: float, recent_c: Sequence[float]) -> float:
    known = [v for v in recent_c if np.isfinite(v)]
    if len(known) < MIN_HOURS:
        return current_c
    return min(current_c, float(np.median(known)))
