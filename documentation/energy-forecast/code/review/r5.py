from ml import *
import numpy as np,pickle
bt=pickle.load(open(d+'/bt.pkl','rb'))
tb=bt['t'];hb=bt['h'];wb=bt['w']
ib={int(x):i for i,x in enumerate(tb)}
idx=np.array([ib.get(int(x),-1) for x in t]);m=idx>=0
Eb=np.full(n,np.nan);Eb[m]=(hb+wb)[idx[m]]
Hb=np.full(n,np.nan);Hb[m]=hb[idx[m]]
ho=set(daylist('ho'));hmask=np.isin(day,list(ho))
# daily comparison between f1 (counter()) and bt (alternative cleaning) on holdout days
dd=sorted(ho);Ef=np.array([E[day==o].sum() for o in dd]);Eo=np.array([np.nansum(Eb[day==o]) for o in dd]);cov=np.array([np.isfinite(Eb[day==o]).mean() for o in dd])
k=cov>.99
print('days with full bt',k.sum(),'/',len(dd),'mean day f1',Ef[k].mean().round(2),'bt',Eo[k].mean().round(2),'mean diff',(Ef-Eo)[k].mean().round(3),'MAD',np.abs(Ef-Eo)[k].mean().round(3),'max',np.abs(Ef-Eo)[k].max().round(2),'corr',np.corrcoef(Ef[k],Eo[k])[0,1].round(5))
print(' days |f1-bt|>1 kWh:',(np.abs(Ef-Eo)[k]>1).sum(),' >3:',(np.abs(Ef-Eo)[k]>3).sum())
worst=np.argsort(-np.abs(Ef-Eo)*k)[:6];print([(str(date.fromordinal(dd[i])),round(Ef[i],1),round(Eo[i],1)) for i in worst])
# COP sanity by day: delivered heat counter / compressor heat kWh
qh=D['qh'];ch=D['ch']
for lbl,rng in (('dev',(DEV0,DEV1)),('ho',(HO[0][0],HO[1][1]))):
    ds=[o for o in range(rng[0],rng[1]) if season(o) and not bad[day==o].any() and (day==o).sum()>=23]
    q=np.array([qh[day==o].sum() for o in ds]);c=np.array([ch[day==o].sum() for o in ds]);T=np.array([Tday[o] for o in ds]);cop=q/np.maximum(c,.5)
    print(lbl,'days',len(ds),'daily COP pct 0,5,50,95,100',np.round(np.nanpercentile(cop,[0,5,50,95,100]),2),'mean T',T.mean().round(1),'COP<1.5:',(cop<1.5).sum(),'COP>6:',(cop>6).sum())
    # COP vs T line: outliers from a local fit
    co=np.polyfit(T,cop,2);r=cop-np.polyval(co,T);print('  COP resid sd',r.std().round(3),'days |r|>3sd',(np.abs(r)>3*r.std()).sum(),[str(date.fromordinal(ds[i])) for i in np.flatnonzero(np.abs(r)>3*r.std())][:12])
# zero-flat runs: exact-zero E with cold weather
print('share of hours exactly 0 by temperature (Oct-Apr, not bad):')
hs=np.array([season(x) for x in day])&~bad
for lo,hi in ((-20,0),(0,5),(5,10),(10,16)):
    k=hs&(Tm>=lo)&(Tm<hi);print(' ',lo,hi,'hours',k.sum(),'CH==0 share',(CH[k]==0).mean().round(3),'E==0',(E[k]==0).mean().round(3))
# runs of zero of length>=4 in cold
zr=np.flatnonzero(hs&(Tm<4)&(CH==0));print('CH==0 hours with T<4:',len(zr));
# Hourly integer-ness: how much of the hourly signal is interpolation
fr=np.mod(E[hs],1);print('share of hours with integer E (1kWh quantised signals):',np.mean(np.isclose(fr,0)|np.isclose(fr,1)).round(3))
# bad hour accounting
for nm,a,b in (('dev',DEV0,DEV1),('ho',HO[0][0],HO[1][1])):
    rg=(day>=a)&(day<b)&hs.__or__(False)
print('counts bad E>10:',(E>10).sum(),'E<0',(E<0).sum(),'NaN',np.isnan(E).sum())
