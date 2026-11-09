from ml import *
import ml,numpy as np,pickle,collections
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365}
ds=daylist('dev');a=actual(ds);h=hwp(ds,28)
L0=run(B,CH,ds);L1=run({**B,"lev":3,"shr":.5},CH,ds);show('dev SIMPLE no lev',score(add(L0,h),a,ds));show('dev SIMPLE lev3^.5',score(add(L1,h),a,ds))
# level correction ending 6h before cut (interpolation sensitivity), dev
def run_gap(cfg,ds,gap=6):
    P={};
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);sv=(ok[c-gap:c].copy());ok[c-gap:c]=False
        P[o]=run(cfg,CH,[o],every=999)[o] if False else None; ok[c-gap:c]=sv
    return P
# implement directly via temporarily masking target just before cut
def lev_gap(cfg,ds,gap):
    P={}
    for o in ds:
        c=idx_cut[o];sv=CH[c-gap:c].copy();CH[c-gap:c]=np.nan
        P[o]=run(cfg,CH,[o])[o];CH[c-gap:c]=sv
    return P
for g in (6,24):
    Lg=lev_gap({**B,"lev":3,"shr":.5},ds,g);show(f'dev SIMPLE last {g}h before cut hidden',score(add(Lg,h),a,ds))
# 95% CI of residual on dR slope (dev, holdout) week-block; causal check
def slope(ds,label):
    act_ch={o:CH[day==o].sum() for o in ds};P={}
    from r8mod import frozen
    return
