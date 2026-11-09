"""The house's fixed physical curves: how fast each zone cools and how it answers the flow temperature.

These are known constants, not fitted by the running system. What the plan may do follows from them;
when we learn more they are changed here, on their own, and the plan follows.

    d(room)/dt = gain * (flow - room) - loss * (room - outdoor)    (per hour)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ZoneCurve:
    name: str
    loss_per_h: float  # cooling per K between the room and outdoors, at the fast end of what was measured
    gain_per_h: (
        float  # how much of the gap to the flow temperature the emitter closes per hour
    )
    reference_c: float  # the room temperature the cooling is judged at


# Ground floor, floor heating. Closed loops at night, 27 Oct - 8 Nov 2026, 5.5-11 C outdoors: 0.09-0.27 C/h,
# median 0.13. The loss is set to the upper part of that (0.19 C/h at 7.5 C). The gain is the measured
# MEASURED_K_SLAB (documentation/ivt-heat-pump.md, "Closed loops measured").
MAIN = ZoneCurve("main", loss_per_h=0.016, gain_per_h=0.0065, reference_c=19.4)
# Salon, radiators. Valves closed, dark hours, Oct 2025 - Nov 2026, 90th percentile of the cooling: 0.29 C/h at
# -4..0 C, 0.17 at 0..4, 0.10 at 4..8. The gain is from the whole-room sensors with the valves open.
SALON = ZoneCurve("salon", loss_per_h=0.0115, gain_per_h=0.0053, reference_c=19.3)
ZONES = (MAIN, SALON)


def cooling_c_per_h(zone: ZoneCurve, room_c: float, t_out_c: float) -> float:
    """How fast the zone falls with no heat coming in (C per hour, never negative)."""
    return max(0.0, zone.loss_per_h * (room_c - t_out_c))


def next_delta_c(
    zone: ZoneCurve, delta_c: float, flow_delta_c: float, open_share: float = 1.0
) -> float:
    """The zone's difference from the flat setpoint one hour on, given the flow's difference from it.

    The flow reaches the room only through what is open: `open_share` of the loops.
    """
    gain = zone.gain_per_h * open_share
    return delta_c + gain * flow_delta_c - (gain + zone.loss_per_h) * delta_c


def next_delta_closed_c(
    zone: ZoneCurve, delta_c: float, open_share: float, gap_c: float
) -> float:
    """As `next_delta_c` for an hour with the loops closed: the heat the flat setpoint would have given is missing.

    `open_share` and `gap_c` are how open the loops would have been and the flow minus the room, under the flat setpoint.
    """
    return delta_c - zone.loss_per_h * delta_c - zone.gain_per_h * open_share * gap_c
