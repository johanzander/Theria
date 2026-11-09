"""What the radiator plan would save.

There is no heat meter per zone, so the figures are estimates: the salon from its thermal model, the
guest house from a crude loss coefficient and labelled indicative.
"""

import math
from collections.abc import Mapping, Sequence

from core.theria.radiators.guest_plan import GuestHour
from core.theria.radiators.salon_plan import SalonHour, SalonModel

SECONDS_PER_HOUR = 3600.0
INDICATIVE = "indicative"


def salon_saving_sek(
    plan: list[SalonHour],
    baseline_c: float,
    t_out_c: Sequence[float],
    model: SalonModel,
    start_mean_c: float | None = None,
) -> float:
    """SEK saved against holding the mean at `baseline_c` all hours (negative: it costs more)."""
    saved = 0.0
    here = baseline_c if start_mean_c is None else start_mean_c
    for hour, t_out in zip(plan, t_out_c, strict=True):
        if not math.isnan(hour.price_sek_per_kwh):
            hold_wh = max(model.u_w_per_k * (baseline_c - t_out), 0.0)
            plan_wh = max(
                model.c_j_per_k * (hour.mean_end_c - here) / SECONDS_PER_HOUR
                + model.u_w_per_k * ((here + hour.mean_end_c) / 2 - t_out),
                0.0,
            )
            saved += hour.price_sek_per_kwh * (hold_wh - plan_wh) / 1000.0
        here = hour.mean_end_c
    return saved


def guest_saving_sek(
    plans: Mapping[str, list[GuestHour]],
    baseline_c: Mapping[str, float],
    prices: Sequence[float],
    loss_w_per_k: float,
) -> tuple[float, str]:
    """An indicative figure: the house's loss shared equally between the rooms, 1 K for an hour."""
    if not plans:
        return 0.0, INDICATIVE
    per_room_w_per_k = loss_w_per_k / len(plans)
    saved = 0.0
    for entity, hours in plans.items():
        for hour, price in zip(hours, prices, strict=True):
            if math.isnan(price):
                continue
            saved += (
                price
                * per_room_w_per_k
                * (baseline_c[entity] - hour.setpoint_c)
                / 1000.0
            )
    return saved, INDICATIVE
