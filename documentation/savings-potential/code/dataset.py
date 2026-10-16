"""Build the hourly study dataset (UTC): spot price, outdoor temp (sensor, gaps from Open-Meteo), sun, measured energy/rts/room."""
import sys, json, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, "/Users/johanzander/GitHub/Theria")
from core.theria.plan.weather import OpenMeteoClient
S = Path(sys.argv[1])
h = np.load(S / "hist.npz")
hours = np.array([dt.datetime.fromtimestamp(x, dt.UTC) for x in h["hours"]])
sp = json.load(open(S / "spot.json"))
price = np.array([sp.get(t.isoformat(), np.nan) for t in hours])
sensor_p = h["price"]
price = np.where(np.isfinite(price), price, sensor_p)
wx = OpenMeteoClient(59.0, 13.0)  # coordinates replaced below
import yaml
from core.theria.plan.settings import PlanSettings
s = PlanSettings.from_options(yaml.safe_load(open("/Users/johanzander/GitHub/Theria/config.yaml"))["options"])
wx = OpenMeteoClient(s.latitude, s.longitude)
w = wx.history(hours[0], hours[-1] + dt.timedelta(hours=1))
wt = {x.start: x.temperature_c for x in w}
om = np.array([wt.get(t, np.nan) for t in hours])
t_s = h["t_out"]
both = np.isfinite(t_s) & np.isfinite(om)
off = float(np.median(t_s[both] - om[both])); r = float(np.corrcoef(t_s[both], om[both])[0, 1])
print("sensor-OpenMeteo offset", off, "corr", r, "rmse", float(np.sqrt(np.mean((t_s[both]-om[both]-off)**2))))
t_all = np.where(np.isfinite(t_s), t_s, om + off)
print("t_out missing after fill:", int((~np.isfinite(t_all)).sum()), "price missing:", int((~np.isfinite(price)).sum()))
np.savez(S / "study.npz", hours=h["hours"], price=price, t_out=t_all, sun=h["cal_sun"],
         rts=h["rts"], comp=h["comp"], aux=h["aux"], hot=h["hot"], room=h["cal_room_ref"], flow=h["cal_flow"])
