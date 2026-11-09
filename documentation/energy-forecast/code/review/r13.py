from ml import *
import ml,numpy as np,pickle
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365,"lev":3,"shr":.5}
def runR(cfg,ds,freeze):
    P={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);sv=R[te].copy()
        if freeze:R[te]=R[c-1]
        P[o]=run(cfg,CH,[o])[o];R[te]=sv
    return P
def slope_ci(ds,lbl):
    Pa=runR(B,ds,False);Pf=runR(B,ds,True);dR=np.array([R[day==o].mean()-R[idx_cut[o]-1] for o in ds]);ac=np.array([np.nansum(CH[day==o]) for o in ds]);rf=np.array([ac[i]-Pf[o].sum() for i,o in enumerate(ds)]);imp=np.array([(Pa[o].sum()-Pf[o].sum()) for o in ds])
    wk=np.array([o//7 for o in ds]);uw=np.unique(wk);rng=np.random.default_rng(0)
    def sl(ix):
        X=np.column_stack([np.ones(len(ix)),dR[ix]]);return np.linalg.lstsq(X,rf[ix],rcond=None)[0][1]
    m=lambda ix:np.median(imp[ix][np.abs(dR[ix])>.2]/dR[ix][np.abs(dR[ix])>.2])
    bs=[]
    for _ in range(600):
        ix=np.concatenate([np.flatnonzero(wk==w) for w in rng.choice(uw,len(uw))]);bs.append((sl(ix),m(ix)))
    bs=np.array(bs);print(lbl,'n',len(ds),'share |dR|>=.75',np.mean(np.abs(dR)>=.75).round(2),'data slope %.2f [%.2f,%.2f] kWh/day/°C ; model implied median %.2f [%.2f,%.2f]'%(sl(np.arange(len(ds))),*np.percentile(bs[:,0],[2.5,97.5]),m(np.arange(len(ds))),*np.percentile(bs[:,1],[2.5,97.5])))
slope_ci(daylist('dev'),'dev');slope_ci(daylist('ho'),'HOLDOUT(diagnostic)')
# forecast with previous_day2 (HOLDOUT sensitivity, not tuning; SIMPLE only)
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
ds=daylist('ho');a=actual(ds);h=hwp(ds,28)
for nm,(tf,sf,wf) in {'pd1':('temperature_2m_previous_day1','shortwave_radiation_previous_day1','wind_speed_10m_previous_day1'),'pd2':('temperature_2m_previous_day2','shortwave_radiation_previous_day2','wind_speed_10m_previous_day2')}.items():
    ml.TF,ml.SF,ml.WF=PRV[tf],PRV[sf],PRV[wf]
    print(nm,'mean abs TF-Tm on holdout hours',np.nanmean(np.abs(ml.TF[np.isin(day,ds)]-Tm[np.isin(day,ds)])).round(2));P=run(B,CH,ds,wx=True);show(f'SIMPLE forecast {nm} [HOLDOUT sensitivity]',score(add(P,h),a,ds))
# cold bias by month, FINAL (HOLDOUT diagnostic)
OUT=pickle.load(open(d+'/h1.pkl','rb'));F=OUT[False]['FINAL mean(linear,gb)+hw28wd'];S=OUT[False]['SIMPLE linear+hw28wd'];ds=daylist('ho');L=pickle.load(open(d+'/review/ho_lin.pkl','rb'))
import collections
g=collections.defaultdict(list)
for o in ds:
    if cold(o):g[date.fromordinal(o).strftime('%Y-%m')].append((F[o].sum()-E[day==o].sum(),S[o].sum()-E[day==o].sum()))
print({k:(len(v),round(np.mean([x[0] for x in v]),1),round(np.mean([x[1] for x in v]),1)) for k,v in sorted(g.items())})
# reasons: DHW error on cold days vs heating
hwf=hwp(ds,28);cd=[o for o in ds if cold(o)];print('cold days: hot water bias',round(np.mean([hwf[o].sum()-CW[day==o].sum() for o in cd]),2),'heating bias SIMPLE',round(np.mean([S[o].sum()-hwf[o].sum()-CH[day==o].sum() for o in cd]),2))
