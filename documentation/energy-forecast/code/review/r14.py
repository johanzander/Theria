from ml import *
import numpy as np,pickle
OUT=pickle.load(open(d+'/h1.pkl','rb'));ds=daylist('ho');a=actual(ds);Pr=pickle.load(open(d+'/price.pkl','rb'));pr=np.array([Pr.get(int(x),np.nan) for x in t],float)
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
# reg28 with the same hot water as FINAL (HOLDOUT, fairness only)
h7=hw(ds,7);h28=hwp(ds,28);R28={o:OUT[False]['reg28+hw7'][o]-h7[o]+h28[o] for o in ds};show('reg28+hw28wd HOLDOUT',score(R28,a,ds))
F=OUT[False]['FINAL mean(linear,gb)+hw28wd'];S=OUT[False]['SIMPLE linear+hw28wd']
de=lambda P:np.array([abs(P[o].sum()-a[o].sum()) for o in ds])
d1=de(F)-de(R28);rng=np.random.default_rng(3);m=len(d1);bs=[]
for _ in range(5000):
    idx=[]
    while len(idx)<m:
        s=rng.integers(0,m);idx+=[(s+j)%m for j in range(14)]
    bs.append(d1[np.array(idx[:m])].mean())
print('FINAL minus reg28+hw28wd: %.2f, block14 95%% [%.2f,%.2f]'%(d1.mean(),*np.percentile(bs,[2.5,97.5])))
# cost: FINAL, measured weather and forecast; cost error as % of total; rolling 30 days
for wx in (False,True):
    P=OUT[wx]['FINAL mean(linear,gb)+hw28wd'];e=[];c=[]
    days=[]
    for o in ds:
        p=pr[day==o]
        if np.isfinite(p).all():e.append((P[o]*p).sum()-(a[o]*p).sum());c.append((a[o]*p).sum());days.append(o)
    e=np.array(e);c=np.array(c);print('wx',wx,'n',len(e),'cost/day',c.mean().round(2),'MAE cost',np.abs(e).mean().round(2),'bias',e.mean().round(3),'%% bias %.2f'%(100*e.sum()/c.sum()),end=' ')
    r30=np.convolve(e,np.ones(30),'valid');t30=np.convolve(c,np.ones(30),'valid');print('30d %% err abs mean %.2f p95 %.2f max %.2f'%(np.mean(np.abs(r30/t30))*100,np.percentile(np.abs(r30/t30),95)*100,np.max(np.abs(r30/t30))*100))
# Hours error vs interpolation: effective noise floor of daily sum due to 1 kWh quantisation, simulated from hourly data
