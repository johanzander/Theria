import requests,pickle,os,numpy as np
from datetime import datetime,timezone
d=os.path.dirname(os.path.abspath(__file__));la,lo=55.578,12.927
def get(url,vars,s,e):
    r=requests.get(url,params=dict(latitude=la,longitude=lo,start_date=s,end_date=e,hourly=",".join(vars),timezone="UTC"),timeout=120);r.raise_for_status();j=r.json()["hourly"]
    t=np.array([int(datetime.fromisoformat(x).replace(tzinfo=timezone.utc).timestamp()) for x in j["time"]]);return t,{v:np.array([np.nan if y is None else y for y in j[v]],float) for v in vars}
end=datetime.now(timezone.utc).strftime("%Y-%m-%d")
a=get("https://archive-api.open-meteo.com/v1/archive",["temperature_2m","shortwave_radiation","wind_speed_10m","relative_humidity_2m","cloud_cover"],"2022-09-25",end)
v=[f"{x}_previous_day{n}" for x in("temperature_2m","shortwave_radiation","wind_speed_10m") for n in(1,2)]
p=get("https://previous-runs-api.open-meteo.com/v1/forecast",v,"2024-02-01",end)
pickle.dump({"archive":a,"prev":p},open(d+"/w1.pkl","wb"))
print(len(a[0]),{k:round(float(np.isfinite(x).mean()),3) for k,x in a[1].items()});print(len(p[0]),{k:round(float(np.isfinite(x).mean()),3) for k,x in p[1].items()})
