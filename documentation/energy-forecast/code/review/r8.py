from ml import *
import ml,numpy as np,pickle
w=lambda ds:hwp(ds,28)
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365,"lev":3,"shr":.5}
def runR(cfg,ds,mode):
    pred={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);sv=R[te].copy()
        if mode=='cut': R[te]=R[c-1]
        if mode=='pre7':R[te]=R[(day>=o-8)&(np.arange(n)<c)].mean()
        pred[o]=run(cfg,CH,[o])[o]
        R[te]=sv
    return pred
for nm in ('dev','ho'):
    ds=daylist(nm);act=actual(ds);hw_=hwp(ds,28);res={}
    for mode in ('actual','cut','pre7'):
        p=runR(B,ds,mode);res[mode]=p;show(f'{nm} SIMPLE R={mode}'+(' [HOLDOUT sensitivity]' if nm=='ho' else ''),score(add(p,hw_),act,ds))
    # natural experiment: days with |R(actual) - R(cut)| >= 0.75 mean
    dR=np.array([R[day==o].mean()-R[idx_cut[o]-1] for o in ds])
    a=np.array([np.nansum(CH[day==o]) for o in ds]);res_cut=np.array([a[i]-res['cut'][o].sum() for i,o in enumerate(ds)]);res_act=np.array([a[i]-res['actual'][o].sum() for i,o in enumerate(ds)])
    sel=np.abs(dR)>=.75;print(' days',len(ds),'with |dR|>=.75:',sel.sum(),' mean |dR| of those',np.abs(dR[sel]).mean().round(2),'std dR all',dR.std().round(2))
    for lbl,s in (('all days',np.ones_like(sel)),('|dR|>=.75',sel)):
        if s.sum()<5:continue
        X=np.column_stack([np.ones(s.sum()),dR[s]]);co=np.linalg.lstsq(X,res_cut[s],rcond=None)[0];co2=np.linalg.lstsq(X,res_act[s],rcond=None)[0]
        print(f'  {lbl}: regress (actual-pred) on dR: model frozen R slope {co[1]:+.2f} kWh/day per °C (implied missing effect); with actual R slope {co2[1]:+.2f}; means |err| frozen {np.abs(res_cut[s]).mean():.2f} actual {np.abs(res_act[s]).mean():.2f}')
    # model implied effect of dR, kWh/day per C
    imp=np.array([(res['actual'][o].sum()-res['cut'][o].sum()) for o in ds]);k=np.abs(dR)>.2
    print('  model implied kWh/day per °C (actual vs frozen):',np.median(imp[k]/dR[k]).round(2),' for T mean',np.mean([Tday[o] for o,kk in zip(ds,k) if kk]).round(1))
