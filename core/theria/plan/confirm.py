"""Waiting for a write to show up when it is read back.

The heat pump takes a few seconds to accept a setpoint over MQTT (Husdata logs "HP Set
confirmed!" three to six seconds after the write) and the thermostats are as slow, so a read
straight after the write sees the old value. Declaring that a failure made the executor think
the setpoint had been changed by hand.
"""

import time
from collections.abc import Callable

CONFIRM_ATTEMPTS = 10  # reads before giving up
CONFIRM_INTERVAL_S = 2.0  # so about 20 seconds in all


def read_until(
    read: Callable[[], float],
    target: float,
    tolerance: float,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Read until the value is within `tolerance` of `target`; the last value read otherwise."""
    value = read()
    for _ in range(CONFIRM_ATTEMPTS - 1):
        if abs(value - target) <= tolerance:
            break
        sleep(CONFIRM_INTERVAL_S)
        value = read()
    return value
