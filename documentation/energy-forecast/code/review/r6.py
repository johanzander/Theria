from ml import *
import numpy as np,pickle
def stats(ds):
    return dict(n=len(ds),T=round(float(np.mean([Tday[o] for o in ds])),2),E=round(float(np.mean([np.nansum(E[day==o]) for o in ds])),1))
for nm,rngs in (('dev',[(DEV0,DEV1)]),('ho',HO)):
    all_=[o for a,b in rngs for o in range(a,b) if season(o)]
    keep=daylist('dev' if nm=='dev' else 'ho');drop=[o for o in all_ if o not in set(keep)]
    print(nm,'all',len(all_),'kept',stats(keep),'dropped',stats(drop))
    import collections
    c=collections.Counter((date.fromordinal(o).strftime('%Y-%m')) for o in drop);print('  drop by month',sorted(c.items()))
    # bad hour share in dropped; can such day have most hours? partial coverage hours
    pc=[bad[day==o].mean() for o in drop];print('  frac bad hrs in dropped: median',np.median(pc).round(2),'share days with <=4 bad hours',np.mean([bad[day==o].sum()<=4 for o in drop]).round(2))
    print('  drop T<3:',sum(1 for o in drop if Tday[o]<3),'of',len(drop),'; keep T<3:',sum(1 for o in keep if Tday[o]<3),'of',len(keep))
# dev days with weird COP
ds=[o for o in range(DEV0,DEV1) if season(o) and not bad[day==o].any() and (day==o).sum()>=23]
for o in ds:
    q=np.nansum(D['qh'][day==o]);c=np.nansum(CH[day==o]);
    if q/max(c,.5)<2 or not np.isfinite(D['qh'][day==o]).all(): print('odd dev',date.fromordinal(o),'qh',round(q,1),'ch',round(c,1),'cw',round(np.nansum(CW[day==o]),1),'T',round(Tday[o],1),'qhfin',np.isfinite(D['qh'][day==o]).mean().round(2))
# how much of holdout total lost? what would FINAL's bias be if dropped days include ones with partial data?
# holdout-specific: score hours on dropped days where E finite (explicitly holdout)
OUT=pickle.load(open(d+'/h1.pkl','rb'))
