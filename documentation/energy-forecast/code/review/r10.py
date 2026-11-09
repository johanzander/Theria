from ml import *
import ml,numpy as np,pickle
# R pattern
print('R by local hour (Oct-Apr 23/24, 24/25, 25/26), mean & std across hours:')
sm=np.array([season(o) for o in day])
for a,b,lbl in ((DEV0,date(2024,5,1).toordinal(),'23/24'),(date(2024,10,1).toordinal(),DEV1,'24/25'),(HO[0][0],HO[0][1],'25/26')):
    k=sm&(day>=a)&(day<b);print(lbl,[round(float(R[k&(hod==h)].mean()),1) for h in range(24)], 'day-to-day std of daily-mean R',np.round(np.std([R[day==o].mean() for o in sorted(set(day[k]))]),2))
# Cold-day bias diagnostics (HOLDOUT diagnostics, not tuning)
OUT=pickle.load(open(d+'/h1.pkl','rb'));days=daylist('ho');act=actual(days)
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365}
L0=run(B,CH,days);L1=run({**B,"lev":3,"shr":.5},CH,days)
G0=run({"gb":1,"hl":365,"leaves":15,"iters":150},CH,days,every=14)
pickle.dump((L0,L1,G0),open(d+'/review/ho_lin.pkl','wb'))
cold_days=[o for o in days if cold(o)]
ch={o:CH[day==o] for o in days}
def b(P,ds):return np.mean([P[o].sum()-ch[o].sum() for o in ds])
for nm,P in (('L0 no level',L0),('L1 level',L1),('G0 gb',G0)):
    print(nm,'cold bias',round(b(P,cold_days),2),'all bias',round(b(P,days),2))
# bias by T bin vs regime
T=np.array([Tday[o] for o in days]);err=np.array([L1[o].sum()-ch[o].sum() for o in days])
for lo,hi in ((-12,0),(0,4),(4,8),(8,12),(12,25)):
    k=(T>=lo)&(T<hi);print(f'T {lo}..{hi}: n {k.sum()}  bias L1 {err[k].mean():+.2f}  MAE {np.abs(err[k]).mean():.2f}  actual CH {np.mean([ch[o].sum() for o,kk in zip(days,k) if kk]):.1f}',end=' | ')
    print('reg28 bias',round(np.mean([OUT[False]["reg28+hw7"][o].sum()-act[o].sum() for o,kk in zip(days,k) if kk]),2))
# saturation? hourly CH on cold days vs max, aux
cs=np.array([np.percentile(CH[(day==o)],95) for o in cold_days]);print('cold days: 95th pct hourly CH',np.round(np.percentile(cs,[10,50,90]),2),' max hourly CH holdout',np.nanmax(CH[np.isin(day,days)]).round(2),'mean aux heating kWh/day cold days',round(np.nanmean([D['ah'][day==o].sum() for o in cold_days]),2),'all days',round(np.nanmean([D['ah'][day==o].sum() for o in days]),2))
# flowhat on cold days
e=[];ek=[]
for o in days:
    c=idx_cut[o];te=np.flatnonzero(day==o);k=np.isfinite(Fa[te])
    if k.any():
        v=(flowhat(te[k],c)-Fa[te[k]]).mean();e.append(v);ek.append(cold(o))
e=np.array(e);ek=np.array(ek);print('flowhat minus Fa day-mean: all',e.mean().round(2),'cold',e[ek].mean().round(2),'non-cold',e[~ek].mean().round(2),'cold n',ek.sum())
# level correction effect on cold days
# compare ratio
print('avg residual prior 3 days vs current residual on cold days: corr',)
r=[];re=[]
for o in cold_days:
    c=idx_cut[o];rec=np.flatnonzero((np.arange(n)<c)&(day>=o-4)&ok&np.isfinite(CH))
    if len(rec)>36:
        pr=L0[o]
    p0=L0[o].sum();p1=L1[o].sum();re.append(p1/p0);r.append(Tday[o]-np.mean([Tday[k] for k in range(o-5,o-1)]))
print('L1/L0 ratio on cold days mean',np.mean(re).round(3),'; T minus prev4 days mean: mean',np.mean(r).round(2))
# In cold days, mean error by bin of how much colder the day is than yesterday (cold onset)
dT=np.array([Tday[o]-Tday[o-1] for o in cold_days]);err2=np.array([L1[o].sum()-ch[o].sum() for o in cold_days]);print('corr(error, dT yesterday-today) cold',np.corrcoef(dT,err2)[0,1].round(2),' mean err when today>=3C colder than yesterday',err2[dT<=-3].mean().round(2),'n',(dT<=-3).sum())
# savings relevance: 30-day rolling sum error
e=np.array([OUT[False]['FINAL mean(linear,gb)+hw28wd'][o].sum()-act[o].sum() for o in days]);tot=np.array([act[o].sum() for o in days]);r30=np.convolve(e,np.ones(30),'valid');t30=np.convolve(tot,np.ones(30),'valid');print('30-day sum error % of total: abs mean',np.mean(np.abs(r30/t30)*100).round(2),'p95',np.percentile(np.abs(r30/t30)*100,95).round(2),'max',np.max(np.abs(r30/t30)*100).round(2))
rc=[];rd=[]
print('cold-day share of total kWh',round(sum(act[o].sum() for o in cold_days)/tot.sum()*100,1),' cold day mean kWh',round(np.mean([act[o].sum() for o in cold_days]),1),' bias%',round(100*np.mean([OUT[False]['FINAL mean(linear,gb)+hw28wd'][o].sum()-act[o].sum() for o in cold_days])/np.mean([act[o].sum() for o in cold_days]),1))
