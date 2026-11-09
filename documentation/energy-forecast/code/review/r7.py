# (a) implied setpoint effect of the frozen SIMPLE/FINAL model; (b) flowhat dF/dR stability; (c) independent daily regression; (d) confounding of R
from ml import *
import numpy as np
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365}
def fit(o,cfg=B):
    c=idx_cut[o];tr=np.flatnonzero((np.arange(n)<c)&ok&np.isfinite(CH));w=0.5**((c-tr)/(24*365))
    F=np.where(np.isfinite(Fa[tr]),Fa[tr],flowhat(tr,c));return ridge(feats(tr,cfg,F),CH[tr],w,1.0),c
def flowco(c):
    m=np.zeros(n,bool);m[max(0,c-60*24):c]=True;m&=np.isfinite(Fa)&~A20
    X=lambda T,r:np.column_stack([np.ones(len(T)),T,T*T,r-18]);co=np.linalg.lstsq(X(Tm[m],R[m]),Fa[m],rcond=None)[0]
    return co,m.sum(),np.std(R[m]),np.ptp(R[m])
days=daylist('ho')+daylist('dev');days=sorted(days)
print('date   n_fit stdR rangeR dF/dR | dCH/dF(h16=0,6,14 kWh/h per K) | dCH/dR (T=10,5,0) kWh/h per °C')
samples=[date(2023,11,20),date(2024,1,15),date(2024,3,15),date(2024,12,1),date(2025,2,15),date(2025,4,1),date(2025,10,20),date(2025,12,15),date(2026,1,15),date(2026,2,15),date(2026,3,15),date(2026,4,15),date(2026,10,20),date(2026,11,1)]
names=None
out=[]
for s in samples:
    o=min(days,key=lambda x:abs(x-s.toordinal()));co,c=fit(o);fc,nm,sr,rr=flowco(c)
    # coefficient positions: intercept, 4 hinges, f, f*h16/10, ...
    fi=1+4; dF=co[fi];dFh=co[fi+1]
    def dch(T):
        h=max(16-T,0);return (dF+dFh*h/10)*fc[3]
    print(date.fromordinal(o),nm,round(sr,2),round(rr,2),round(fc[3],2),'|',round(dF,3),round(dF+dFh*.6,3),round(dF+dFh*1.4,3),'|',round(dch(10),3),round(dch(5),3),round(dch(0),3))
    out.append((dch(10),dch(5),dch(0)))
out=np.array(out);print('implied kWh/h per °C of setpoint, mean',out.mean(0).round(3),'min',out.min(0).round(3),'max',out.max(0).round(3))
# (c) GB finite difference of R on holdout-ish days: FINAL model's R response with flowhat (R +1) vs base (uses dev fit only date 2025-03-01 - no holdout data)
o=min(daylist('dev'),key=lambda x:abs(x-date(2025,3,1).toordinal()));c=idx_cut[o];te=np.flatnonzero((day>=o-30)&(day<=o))
tr=np.flatnonzero((np.arange(n)<c)&ok&np.isfinite(CH));w=0.5**((c-tr)/(24*365));F=np.where(np.isfinite(Fa[tr]),Fa[tr],flowhat(tr,c))
from sklearn.ensemble import HistGradientBoostingRegressor as GB
g=GB(max_iter=150,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,monotonic_cst=[-1,-1,-1,-1,0,0,0,0,0,0],random_state=0).fit(gbX(tr,F),CH[tr],sample_weight=w)
sv=R.copy()
p0=g.predict(gbX(te,flowhat(te,c)));R[te]=R[te]+1.0;p1=g.predict(gbX(te,flowhat(te,c)));R[:]=sv
print('GB finite diff: dCH per +1°C setpoint on',len(te),'hours T',Tm[te].mean().round(1),':',(p1-p0).mean().round(3),'kWh/h')
