"""Fetch history + fit models, cache to npz/json (read-only against InfluxDB)."""
import sys, json, pickle
from datetime import UTC, datetime, timedelta
from pathlib import Path
import numpy as np, yaml
from dotenv import load_dotenv
sys.path.insert(0, "/Users/johanzander/GitHub/Theria")
load_dotenv("/Users/johanzander/GitHub/Theria/.env")
from core.theria.plan.calibration import Calibrator
from core.theria.plan.calibration_data import InfluxHistorySource, load_calibration_inputs
from core.theria.plan.outcomes import load_outcome_inputs
from core.theria.plan.settings import PlanSettings
from core.theria.plan.weather import OpenMeteoClient

out = Path(sys.argv[1])
options = yaml.safe_load(Path("/Users/johanzander/GitHub/Theria/config.yaml").read_text())["options"]
s = PlanSettings.from_options(options)
src = InfluxHistorySource()
now = datetime.now(UTC)
wx = OpenMeteoClient(s.latitude, s.longitude)
models = Calibrator(src, wx, s, days=1460).run(now)
print(models)
oi = load_outcome_inputs(src, s, now - timedelta(days=1460), now)
ci = load_calibration_inputs(src, wx, s, now, 1460)
np.savez(out / "hist.npz",
  hours=np.array([h.timestamp() for h in oi.hours]),
  price=oi.price_sek_per_kwh, rts=oi.rts_c, t_out=oi.t_out_c,
  comp=oi.compressor_kwh, aux=oi.aux_kwh, hot=oi.hot_water, room=oi.room_c,
  cal_hours=np.array([h.timestamp() for h in ci.hours]), cal_sun=ci.sun_wm2,
  cal_room_ref=ci.room_reference_c, cal_flow=ci.flow_c, cal_tout=ci.t_out_c)
pickle.dump({"pump": models.pump.params, "room": models.room.params if models.room else None,
             "night": models.night_drift, "settings": s}, open(out / "models.pkl", "wb"))
