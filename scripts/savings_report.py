"""What did the heating cost, and what did the setpoint control save? (read-only)

Run from the repository root:  .venv/bin/python scripts/savings_report.py [days_back] [base]

`base` is the base setpoint (default: the 7-day running median of the setpoint, which suits
the years of manual control). Uses the pump model fitted on the history, the pump's own
electricity counters, and the hourly price.
"""

import sys
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.theria.plan.calibration import fit_pump
from core.theria.plan.calibration_data import (
    InfluxHistorySource,
    load_calibration_inputs,
)
from core.theria.plan.outcomes import day_outcomes, load_outcome_inputs
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import OpenMeteoClient


def running_median(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for i in range(len(values)):
        w = values[max(0, i - window) : i + 1]
        w = w[np.isfinite(w)]
        if len(w):
            out[i] = np.median(w)
    return out


def main() -> None:
    load_dotenv()
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 1460
    options = yaml.safe_load(Path("config.yaml").read_text())["options"]
    settings = PlanSettings.from_options(options)
    assert settings is not None
    source = InfluxHistorySource()
    now = datetime.now(UTC)
    weather = OpenMeteoClient(settings.latitude, settings.longitude)
    pump = fit_pump(
        load_calibration_inputs(source, weather, settings, now, 1460)
    ).params
    inputs = load_outcome_inputs(source, settings, now - timedelta(days=days_back), now)
    base = (
        float(sys.argv[2])
        if len(sys.argv) > 2
        else running_median(inputs.rts_c, 7 * 24)
    )
    outcomes = day_outcomes(inputs, pump, base)
    months: dict[str, list] = defaultdict(list)
    for o in outcomes:
        if o.heating_hours >= 12:
            months[o.day.strftime("%Y-%m")].append(o)
    print(
        "month   days  cost(SEK)  baseline  saving   saving%  price paid/baseline  aux kWh  compressor-stopped h"
    )
    total_actual = total_base = 0.0
    for month, items in sorted(months.items()):
        a = sum(o.actual_cost_sek for o in items)
        b = sum(o.baseline_cost_sek for o in items)
        paid = np.mean([o.average_price_paid_sek_per_kwh for o in items])
        basep = np.mean([o.average_price_baseline_sek_per_kwh for o in items])
        total_actual += a
        total_base += b
        print(
            f"{month}  {len(items):3d}  {a:9.0f}  {b:8.0f}  {b - a:7.0f}  {100 * (b - a) / b if b else 0:6.1f}%   "
            f"{paid:5.2f} / {basep:5.2f}      {sum(o.aux_kwh for o in items):6.0f}   "
            f"{sum(o.compressor_stopped_hours for o in items):5d}"
        )
    print(
        f"TOTAL cost {total_actual:.0f}, baseline {total_base:.0f}, saving {total_base - total_actual:.0f} SEK"
    )


if __name__ == "__main__":
    main()
