"""Fit the plan models on real history and print the result (read-only).

Run from the repository root:  .venv/bin/python scripts/calibration_report.py [days]
Needs .env with HA_DB_URL / HA_DB_USER_NAME / HA_DB_PASSWORD (InfluxDB) and the
`plan` section of config.yaml.
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import InfluxHistorySource
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import OpenMeteoClient


def main() -> None:
    load_dotenv()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1460
    options = yaml.safe_load(Path("config.yaml").read_text())["options"]
    settings = PlanSettings.from_options(options)
    if settings is None:
        raise SystemExit("config.yaml has no plan section")
    calibrator = Calibrator(
        InfluxHistorySource(),
        OpenMeteoClient(settings.latitude, settings.longitude),
        settings,
        days=days,
    )
    fitted = calibrator.run(datetime.now(UTC))
    print(fitted)


if __name__ == "__main__":
    main()
