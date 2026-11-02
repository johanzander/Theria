"""The guest house: a setpoint per room that moves within a band around the room's own temperature.

The cheapest hour of the horizon gets the top of the band, the dearest the bottom; the rank is the
price's place among the known prices, so the plan needs no price level to compare with. A room that is
already outside its band is never pushed further out.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from core.theria.radiators.settings import GuestRoom

HALF = 0.5
FULL_SPREAD_SEK_PER_KWH = 0.2  # the price spread at which the whole band is used


@dataclass(frozen=True)
class GuestHour:
    start: datetime
    setpoint_c: float
    reason: str


def _round_half(value: float) -> float:
    return math.floor(value / HALF + 0.5) * HALF


def _rank(price: float, known: Sequence[float]) -> float:
    """0 for the cheapest price, 1 for the dearest; ties share their mean rank."""
    if len(known) < 2:
        return 0.5
    less = sum(p < price for p in known)
    equal = sum(p == price for p in known)
    return (less + (equal - 1) / 2) / (len(known) - 1)


def plan_guest_room(
    hours: list[datetime],
    prices: Sequence[float],
    room: GuestRoom,
    band_c: float,
    room_temp_c: float | None,
    min_c: float = 5.0,
    max_c: float = 25.0,
) -> list[GuestHour]:
    known = [p for p in prices if not math.isnan(p)]
    # near-flat prices give near-flat setpoints: the band is used in proportion to the spread
    reach = (
        min(1.0, (max(known) - min(known)) / FULL_SPREAD_SEK_PER_KWH) if known else 0.0
    )
    out: list[GuestHour] = []
    for start, price in zip(hours, prices, strict=True):
        if math.isnan(price):
            setpoint, reason = room.t_room_c, "no price, kept at the room temperature"
        else:
            f = (1 - 2 * _rank(price, known)) * reach
            setpoint = room.t_room_c + band_c * f
            reason = f"price rank {_rank(price, known):.2f} of the horizon"
            if room_temp_c is not None:
                if room_temp_c < room.t_room_c - band_c and setpoint < room.t_room_c:
                    setpoint = room.t_room_c
                    reason = "room below its band, not cut"
                elif room_temp_c > room.t_room_c + band_c and setpoint > room.t_room_c:
                    setpoint = room.t_room_c
                    reason = "room above its band, not heated further"
        out.append(
            GuestHour(start, min(max_c, max(min_c, _round_half(setpoint))), reason)
        )
    return out
