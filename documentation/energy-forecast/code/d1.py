import numpy as np,pickle,os,warnings,requests
warnings.filterwarnings("ignore")
from datetime import datetime,timezone
from core.theria.influxdb_helper import query_influx as Q
d=os.path.dirname(os.path.abspath(__file__));H=3600
A=int(datetime(2022,10,1,tzinfo=timezone.utc).timestamp());B=int(datetime.now(timezone.utc).timestamp())//H*H
grid=np.arange(A,B+H,H)
def counter(m,maxrate=7.5):
    r=Q([f'SELECT "value" FROM "{m}" WHERE time>={A}s AND time<{B}s'])[0]
    t=np.array([x[0]/1000 for x in r]);v=np.array([x[1] for x in r],float)
    o=np.argsort(t);t,v=t[o],v[o]
    kt=[t[0]];kc=[0.];ref=v[0];shifts=gl=0
    for i in range(1,len(v)):
        dv=v[i]-ref;lim=2+maxrate*(t[i]-kt[-1])/H
        if 0<=dv<=lim:
            kt.append(t[i]);kc.append(kc[-1]+dv);ref=v[i]
        else:
            nx=v[i+1:i+5]
            if len(nx)>=3 and np.all(np.abs(nx-v[i])<=8) and np.all(np.diff(nx)>=-1e-9):
                ref=v[i];shifts+=1;kt.append(t[i]);kc.append(kc[-1])
            else: gl+=1
    kt,kc=np.array(kt),np.array(kc)
    c=np.interp(grid,kt,kc);e=np.diff(c)
    # hours inside a gap > 8 h between readings are unknown
    idx=np.searchsorted(kt,grid[:-1]+H/2);idx=np.clip(idx,1,len(kt)-1);gap=kt[idx]-kt[idx-1];e[gap>8*H]=np.nan
    e[(grid[:-1]<kt[0])|(grid[:-1]>kt[-1])]=np.nan
    print(m,'raw',len(t),'kept',len(kt),'shifts',shifts,'glitch',gl,'sum',round(np.nansum(e)),'nan%',round(100*np.isnan(e).mean(),1),'max',round(np.nanmax(e),1))
    return e
def agg(m,f,fn,lo=None,hi=None):
    c=f' AND "{f}">={lo} AND "{f}"<={hi}' if lo is not None else ""
    r=Q([f'SELECT {fn}("{f}") FROM "{m}" WHERE time>={A}s AND time<{B}s{c} GROUP BY time(1h) fill(none)'])[0]
    e={x[0]//1000:x[1] for x in r};return np.array([e.get(int(t),np.nan) if e.get(int(t)) is not None else np.nan for t in grid[:-1]],float)
D=dict(t=grid[:-1],
 ch=counter("sensor.compr_cons_heating"),cw=counter("sensor.compr_cons_hotwat"),
 ah=counter("sensor.aux_cons_heating",10),aw=counter("sensor.aux_cons_hot_water",10),
 qh=counter("sensor.supp_energy_heating",30),
 O=agg("sensor.outdoor","value","mean",-35,45),
 F=agg("sensor.heating_setpoint","value","last",15,70),
 R=agg("climate.room_temp_setpoint","temperature","last",5,30),
 V=agg("binary_sensor.switch_valve_1","value","mean",0,1),
 LV=agg("climate.lk_vardagsrum_thermostat","temperature","last",5,30),LB=agg("climate.lk_bibliotek_thermostat","temperature","last",5,30),
 RV=agg("climate.lk_vardagsrum_thermostat","current_temperature","mean",5,35),RB=agg("climate.lk_bibliotek_thermostat","current_temperature","mean",5,35),
 S1=agg("sensor.sonoff_a480045bf5_temperature","value","mean",5,35))
pickle.dump(D,open(d+"/f1.pkl","wb"))
def ym(x,m):
    return x[m]
for k,v in D.items(): print(k,round(float(np.isfinite(v).mean()),3))
