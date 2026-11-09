# poison everything at/after the cut and check predictions for sampled days are unchanged
from ml import *
import ml, numpy as np
ALL=[o for o in daylist('dev')[::15]+daylist('ho')[::12]]
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365,"lev":3,"shr":.5}
GB={"gb":1,"hl":365,"leaves":15,"iters":60,"lev":3,"shr":.5}
def poison(c,te,wxmode):
    sv={}
    arrs={'CH':CH,'CW':CW,'E':E,'Fa':Fa,'qh':D['qh']}
    for k,a in arrs.items():
        sv[k]=a[c:].copy()
        a[c:]=a[c:]*7+13
    if wxmode:
        for nm,a in (('Tm',Tm),('Sun',Sun),('Wind',Wind)):
            sv[nm]=a[c:te[-1]+1].copy(); a[c:te[-1]+1]+=11
    return sv
def unpoison(c,te,sv,wxmode):
    for k,a in (('CH',CH),('CW',CW),('E',E),('Fa',Fa),('qh',D['qh'])): a[c:]=sv[k]
    if wxmode:
        for nm,a in (('Tm',Tm),('Sun',Sun),('Wind',Wind)): a[c:te[-1]+1]=sv[nm]
res=[]
for wx in (False,True):
    for name,cfg in (('lin',B),('gb',GB)):
        d0=[];
        for o in ALL:
            if wx and not (np.isfinite(TF[idx_cut[o]:np.flatnonzero(day==o)[-1]+1])).all(): continue
            if name=='gb' and ALL.index(o)%3: continue
            c=idx_cut[o];te=np.flatnonzero(day==o)
            a=run(cfg,CH,[o],wx=wx)[o]; hwa=hw([o],28)[o]
            sv=poison(c,te,wx); b=run(cfg,CH,[o],wx=wx)[o]; hwb=hw([o],28)[o]; unpoison(c,te,sv,wx)
            d0.append((np.abs(a-b).max(),np.abs(hwa-hwb).max()))
        d0=np.array(d0);print('wx',wx,name,'days',len(d0),'max|Δpred|',d0[:,0].max(),'max|Δhw|',d0[:,1].max(),flush=True)
