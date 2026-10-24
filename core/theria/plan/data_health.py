"""Are the inputs the plan and the savings depend on there, and sane? Checked against what Home Assistant has stored.

The plan reads today's and tomorrow's prices from the Nord Pool feed and the room, the pump and the weather from
live states. The measured savings read the prices of the finished days from the same feed and the stored history of
the pump's two energy counters, the setpoint, the weather and the room. A silent sensor or a replaced entity leaves those
without data for days and nothing else complains, so each input is checked on its own and the result is shown next to the
other checks.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from core.theria.exceptions import PriceDataError
from core.theria.plan.calibration_data import FLOW_PLAUSIBLE_C
from core.theria.plan.outcomes import OutcomeInputs
from core.theria.plan.preflight import READ_ERRORS, Check, StateClient
from core.theria.plan.prices import PricePeriod, hourly_prices
from core.theria.plan.series import FloatArray, hour_floor
from core.theria.plan.settings import PlanSettings

WINDOW_HOURS = 24  # how far back each input is looked at
COUNTED_WINDOW_HOURS = 72
OK_SHARE = 0.9
WARN_SHARE = 0.5
MIN_PRICE_HOURS_AHEAD = 6  # the plan needs a few hours of prices to move anything
TOMORROW_PUBLISHED_HOUR = 14  # local time after which tomorrow's prices should be there


def _share(values: FloatArray) -> float:
    return float(np.mean(np.isfinite(values))) if len(values) else 0.0


def history_checks(inputs: OutcomeInputs, s: PlanSettings) -> list[Check]:
    """For each stored input: the share of the last day's hours that have a value."""
    specs = [
        ("price history", "the Nord Pool feed", inputs.price_sek_per_kwh),
        ("outdoor history", s.outdoor_entity, inputs.t_out_c),
        ("pump setpoint history", s.room_setpoint_entity, inputs.rts_c),
        ("compressor counter", s.energy_counter_entity, inputs.compressor_kwh),
        ("electric heater counter", s.aux_energy_entity, inputs.aux_kwh),
        ("room sensors", ", ".join(s.room_temp_entities), inputs.room_c),
    ]
    checks = []
    for name, entity, series in specs:
        share = _share(series[-WINDOW_HOURS:])
        detail = f"{entity}: {share:.0%} of the last {WINDOW_HOURS} h"
        if share >= OK_SHARE:
            checks.append(Check(name, "ok", detail))
        elif share > 0:
            checks.append(Check(name, "warn", detail))
        else:
            checks.append(Check(name, "warn", f"{entity}: NO DATA in the last {WINDOW_HOURS} h"))
    return checks


def counted_hours_check(inputs: OutcomeInputs, s: PlanSettings) -> Check:
    """The measured savings count the heating-only hours that have every input; which input costs most hours?"""
    window = slice(-COUNTED_WINDOW_HOURS, None)
    heating = inputs.hot_water[window] == 0.0
    if not heating.any():
        return Check("measured savings", "warn", "no heating-only hours in the last 3 days")
    series = {
        "price history": inputs.price_sek_per_kwh[window],
        "outdoor": inputs.t_out_c[window],
        "pump setpoint": inputs.rts_c[window],
        "compressor counter": inputs.compressor_kwh[window],
        "electric heater counter": inputs.aux_kwh[window],
    }
    counted = heating.copy()
    for values in series.values():
        counted &= np.isfinite(values)
    share = float(counted.sum() / heating.sum())
    detail = f"{int(counted.sum())} of {int(heating.sum())} heating-only hours in the last 3 days count ({share:.0%})"
    if share >= 0.8:
        return Check("measured savings", "ok", detail)
    worst_name, worst_share = min(
        ((name, _share(values[heating])) for name, values in series.items()), key=lambda item: item[1]
    )
    return Check("measured savings", "warn", f"{detail}; fewest values: {worst_name} ({worst_share:.0%})")


def price_feed_check(periods: list[PricePeriod], now: datetime, timezone: str) -> Check:
    """The prices the plan steers on: enough hours ahead, and tomorrow's once they are published."""
    try:
        hours = hourly_prices(periods, hour_floor(now))
    except PriceDataError as e:
        return Check("price feed", "fail", f"no prices to plan with: {e}")
    last_end = hours[-1][0] + timedelta(hours=1)
    ahead = len(hours)
    local = now.astimezone(ZoneInfo(timezone))
    tomorrow_end = (local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2))
    detail = f"{ahead} h of prices ahead, to {last_end.astimezone(ZoneInfo(timezone)):%a %H:%M}"
    if ahead < MIN_PRICE_HOURS_AHEAD:
        return Check("price feed", "fail", f"only {detail}")
    if local.hour >= TOMORROW_PUBLISHED_HOUR and last_end.astimezone(ZoneInfo(timezone)) < tomorrow_end:
        return Check("price feed", "warn", f"{detail}; tomorrow's prices are not here yet")
    return Check("price feed", "ok", detail)


def live_h60_checks(ha: StateClient, s: PlanSettings) -> list[Check]:
    """The H60 answers 0 or nonsense now and then: a flow reading outside what a working pump gives is flagged."""
    low, high = FLOW_PLAUSIBLE_C
    checks = []
    for name, entity in (("flow temperature", s.flow_measured_entity), ("flow target", s.flow_setpoint_entity)):
        try:
            value = float(ha.get_state(entity)["state"])
        except READ_ERRORS as e:
            checks.append(Check(name, "warn", f"cannot read {entity}: {e}"))
            continue
        if low <= value <= high:
            checks.append(Check(name, "ok", f"{entity}: {value:.1f} C"))
        else:
            checks.append(Check(name, "warn", f"{entity}: {value:.1f} C is outside {low:.0f}-{high:.0f} C (an H60 glitch?)"))
    return checks
