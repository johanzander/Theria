"""Walk-forward backtest library for pump electricity. Forecast of local day D is issued 14:00 local on D-1."""
import numpy as np,pickle,os,warnings
from datetime import datetime,date,timedelta
from zoneinfo import ZoneInfo
warnings.filterwarnings("ignore")
L=ZoneInfo("Europe/Stockholm");d=os.path.dirname(os.path.abspath(__file__))
D=pickle.load(open(d+"/f1.pkl","rb"));W=pickle.load(open(d+"/w1.pkl","rb"))
t=D["t"];n=len(t);loc=[datetime.fromtimestamp(int(x),L) for x in t]
hod=np.array([x.hour for x in loc]);dow=np.array([x.weekday() for x in loc]);day=np.array([x.date().toordinal() for x in loc]);doy=np.array([x.timetuple().tm_yday for x in loc])
def align(ts,v):
    m=dict(zip(ts.tolist(),v.tolist()));return np.array([m.get(int(x),np.nan) for x in t],float)
ERA={k:align(W["archive"][0],v) for k,v in W["archive"][1].items()}
PRV={k:align(W["prev"][0],v) for k,v in W["prev"][1].items()}
def ffill(x,lim):
    o=x.copy();last=np.nan;age=10**9
    for i in range(len(o)):
        if np.isfinite(o[i]): last=o[i];age=0
        else:
            age+=1
            if age<=lim:o[i]=last
    return o
# outdoor: sensor, gaps filled by ERA5 shifted by the mean difference
O=D["O"].copy();b=np.nanmean(O-ERA["temperature_2m"]);print("sensor-ERA5 bias",round(float(b),2));Tm=np.where(np.isfinite(O),O,ERA["temperature_2m"]+b)
R=ffill(D["R"],24*400);R=np.where(np.isfinite(R),R,20.0)
Sun=ERA["shortwave_radiation"];Wind=ERA["wind_speed_10m"]
CH=D["ch"];CW=D["cw"];E=CH+CW
bad=~np.isfinite(E)|(E<0)|(E>10)
A20=(t>=datetime(2026,11,8,12,41,tzinfo=L).timestamp())&(t<datetime(2026,11,9,17,25,tzinfo=L).timestamp())
def trail(x,w):
    c=np.cumsum(np.nan_to_num(x));out=np.empty(n);out[:]=np.nan
    out[w:]=(c[w:]-c[:-w])/w;out[:w]=np.nanmean(x[:w]);return out
DEV0=date(2023,10,1).toordinal();DEV1=date(2025,5,1).toordinal();HO=[(date(2025,10,1).toordinal(),date(2026,5,1).toordinal()),(date(2026,10,1).toordinal(),date(2026,11,9).toordinal())]
def season(o):
    x=date.fromordinal(o);return x.month in (10,11,12,1,2,3,4)
def daylist(kind):
    out=[]
    if kind=="dev":rng=[(DEV0,DEV1)]
    else:rng=HO
    for a,b in rng:
        for o in range(a,b):
            if season(o):
                m=day==o
                if m.sum()>=23 and not bad[m].any():out.append(o)
    return out
idx_cut={}
for o in sorted(set(day)):
    x=np.flatnonzero((day==o-1)&(hod>=14))
    idx_cut[o]=x[0] if len(x) else None
Tday={o:Tm[day==o].mean() for o in set(day)}
def cold(o):
    p=[Tday[k] for k in range(o-29,o-1) if k in Tday];return len(p)>10 and Tday[o]<=np.mean(p)-4
def score(pred,act,days):
    """pred, act: dict day->24ish arrays"""
    he=[];de=[];w4=[];cd=[];cw=[]
    for o in days:
        p,a=pred[o],act[o];he+=list(np.abs(p-a));e=p.sum()-a.sum();de.append(e)
        k=len(a)//4;w4+=[abs(p[i:i+4].sum()-a[i:i+4].sum()) for i in range(0,4*k,4)]
        if cold(o):cd.append(e)
    de=np.array(de);a_=sum(act[o].sum() for o in days)
    return dict(dayMAE=np.abs(de).mean(),bias=de.mean(),pct=100*np.abs(de).sum()/a_,w4=np.mean(w4),hour=np.mean(he),coldMAE=np.abs(cd).mean() if cd else np.nan,coldbias=np.mean(cd) if cd else np.nan,ncold=len(cd),n=len(days))
def show(name,s):
    print(f"{name:42s} day {s['dayMAE']:5.2f} bias {s['bias']:+5.2f} {s['pct']:4.1f}% | 4h {s['w4']:.2f} hr {s['hour']:.3f} | cold {s['coldMAE']:5.2f} ({s['coldbias']:+5.2f}) n{s['ncold']}/{s['n']}",flush=True)
def actual(days):
    return {o:E[day==o] for o in days}
