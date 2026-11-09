# causal check: independent daily regression + season FE; flow-only claim (DEV+2022/23 only)
from ml import *
import numpy as np
yr=lambda o:(lambda x:x.year if x.month>=7 else x.year-1)(date.fromordinal(o))
seasons=[o for o in sorted(set(day)) if season(o) and (day==o).sum()>=23 and not bad[day==o].any()]
Sdev=[o for o in seasons if o<HO[0][0]]
d_=np.array(Sdev);y=np.array([CH[day==o].sum() for o in Sdev]);T=np.array([Tday[o] for o in Sdev]);RR=np.array([R[day==o].mean() for o in Sdev]);sn=np.array([yr(o) for o in Sdev]);hS=np.array([Sun[day==o].mean() for o in Sdev])/100;wnd=np.array([Wind[day==o].mean() for o in Sdev])
T1=np.array([Tday.get(o-1,np.nan) for o in Sdev]);T3=np.array([np.mean([Tday.get(o-k,np.nan) for k in (1,2,3)]) for o in Sdev])
ok1=np.isfinite(T1)&np.isfinite(T3)
hd=np.maximum(16-T,0)
def fit(names):
    X=np.column_stack(names);X=X[ok1];co=np.linalg.lstsq(X,y[ok1],rcond=None)[0];r=y[ok1]-X@co;se=np.sqrt(np.sum(r**2)/(len(r)-X.shape[1])*np.linalg.pinv(X.T@X).diagonal());return co,se,r
base=[np.ones(len(y)),hd,hd**2,np.maximum(16-T1,0),np.maximum(16-T3,0),hS,wnd]
# With R alone, with R x hd
c1,s1,r1=fit(base+[RR-20]);i=len(base)
print('no season FE: dE/dR = %.2f ± %.2f kWh/day per °C (R std %.2f)'%(c1[i],s1[i],RR[ok1].std()))
FE=[(sn==s).astype(float) for s in sorted(set(sn))][1:]
c2,s2,r2=fit(base+FE+[RR-20]);i2=len(base)+len(FE)
print('season FE (within-winter): dE/dR = %.2f ± %.2f kWh/day per °C; residual sd %.2f vs %.2f'%(c2[i2],s2[i2],r2.std(),r1.std()),' season FE coefs',np.round(c2[len(base):i2],1),'; R mean by season',[ (s,RR[sn==s].mean().round(2)) for s in sorted(set(sn))])
# R x cold
c3,s3,r3=fit(base+FE+[RR-20,(RR-20)*hd/10]);i3=len(base)+len(FE)
print('with Rxhd season FE: dE/dR %.2f  + hd/10*%.2f ; at T=10: %.2f, T=0: %.2f kWh/day/°C  = %.3f, %.3f kWh/h/°C'%(c3[i3],c3[i3+1],c3[i3]+c3[i3+1]*.6,c3[i3]+c3[i3+1]*1.6,(c3[i3]+c3[i3+1]*.6)/24,(c3[i3]+c3[i3+1]*1.6)/24))
# autocorr robust: weekly block bootstrap of the within-season slope
rng=np.random.default_rng(0);wk=np.array([o//7 for o in d_[ok1]]);uw=np.unique(wk);Xall=np.column_stack(base+FE+[RR-20])[ok1];yy=y[ok1];bs=[]
for _ in range(1500):
    ix=np.concatenate([np.flatnonzero(wk==w) for w in rng.choice(uw,len(uw))]);c=np.linalg.lstsq(Xall[ix],yy[ix],rcond=None)[0];bs.append(c[-1])
print('week-block bootstrap dE/dR within-season: 95%%CI [%.2f,%.2f] kWh/day/°C'%tuple(np.percentile(bs,[2.5,97.5])))
# within-day: does R (F-residual) explain hour CH within day? (quick)
# Flow-only claim with TRUE flow on hours where flow measured, DEV hours
idx=np.flatnonzero(np.isin(day,Sdev)&ok&np.isfinite(Fa)&np.isfinite(CH));X=np.column_stack([np.ones(len(idx)),Fa[idx],Fa[idx]**2]);co=np.linalg.lstsq(X,CH[idx],rcond=None)[0];f1=X@co
Xh=np.column_stack([np.ones(len(idx))]+hinges(Tm[idx],(16,10,4,-2)));coh=np.linalg.lstsq(Xh,CH[idx],rcond=None)[0];f2=Xh@coh
Xb=np.column_stack([Xh,Fa[idx]-35,(Fa[idx]-35)*np.maximum(16-Tm[idx],0)/10]);cob=np.linalg.lstsq(Xb,CH[idx],rcond=None)[0];f3=Xb@cob
print('hours w/ real flow (dev+22/23) n',len(idx),'R2: flow-only %.3f, T hinges %.3f, hinges+flow %.3f; corr(Fa,hinges T fit)=%.3f'%(1-((CH[idx]-f1)**2).sum()/((CH[idx]-CH[idx].mean())**2).sum(),1-((CH[idx]-f2)**2).sum()/((CH[idx]-CH[idx].mean())**2).sum(),1-((CH[idx]-f3)**2).sum()/((CH[idx]-CH[idx].mean())**2).sum(),np.corrcoef(Fa[idx],f2)[0,1]))
# flowhat accuracy: rmse of flow from T,R on dev ... with F logged only on change?
fin=np.isfinite(D['F']);print('F logged hours share',fin.mean().round(3),'; median gap between F hours (h, over Oct-Apr):',np.median(np.diff(np.flatnonzero(fin&np.array([season(o) for o in day])))))
idx=np.flatnonzero(np.isin(day,Sdev)&np.isfinite(D['F']));c=idx_cut[Sdev[-1]];e=flowhat(idx,c)-D['F'][idx];print('flowhat vs logged F on dev (last-window fit, in-sample flavour): MAE',np.abs(e).mean().round(2))
# variance of F not explained by T,R
X=np.column_stack([np.ones(len(idx)),Tm[idx],Tm[idx]**2,R[idx]-18]);co=np.linalg.lstsq(X,D['F'][idx],rcond=None)[0];r=D['F'][idx]-X@co;print('global OLS F~T,T2,R on dev: R2 %.3f, resid sd %.2f, coef R %.2f'%(1-r.var()/D['F'][idx].var(),r.std(),co[3]))
