from bm import *
from sklearn.ensemble import HistGradientBoostingRegressor as GB
Fa=ffill(D["F"],6);T24=trail(Tm,24);T72=trail(Tm,72);T168=trail(Tm,168);ok=~bad&~A20
def hinges(T,knots):return [np.maximum(k-T,0) for k in knots]
def flowhat(i,cut):
    """Flow target for rows i from a fit on the last 60 days before cut (the pump's curve as it stands)."""
    m=np.zeros(n,bool);m[max(0,cut-60*24):cut]=True;m&=np.isfinite(Fa)&~A20
    if m.sum()<200:m=np.isfinite(Fa)&(np.arange(n)<cut)
    X=lambda T,r:np.column_stack([np.ones(len(T)),T,T*T,r-18])
    co=np.linalg.lstsq(X(Tm[m],R[m]),Fa[m],rcond=None)[0];return X(Tm[i],R[i])@co
def feats(i,cfg,F):
    T=Tm[i];h16=np.maximum(16-T,0);c=[np.ones(len(i))]+hinges(T,cfg.get("knots",(16,10,4,-2)))
    if cfg.get("rts"):c+= [R[i]-18,(R[i]-18)*h16/10]
    if cfg.get("flow"):f=F-35;c+=[f,f*h16/10]
    if cfg.get("lag"):c+=[np.maximum(16-T24[i],0),np.maximum(16-T72[i],0)]
    if cfg.get("lag7"):c+=[np.maximum(16-T168[i],0)]
    if cfg.get("sun"):c+=[Sun[i]/100,Sun[i]/100*h16/10]
    if cfg.get("wind"):c+=[Wind[i]/10*h16/10]
    if cfg.get("hod"):
        H=np.eye(24)[hod[i]][:,1:];c+=list(H.T)
    if cfg.get("hodh"):
        H=np.eye(24)[hod[i]][:,1:];c+=list((H.T*h16/10))
    if cfg.get("dow"):
        c+=list(np.eye(7)[dow[i]][:,1:].T)
    return np.column_stack(c)
def ridge(X,y,w,l):
    W=w[:,None];A=X.T@(X*W);A+=l*np.eye(X.shape[1]);A[0,0]-=l;return np.linalg.solve(A,X.T@(y*w))
TF,SF,WF=PRV["temperature_2m_previous_day1"],PRV["shortwave_radiation_previous_day1"],PRV["wind_speed_10m_previous_day1"]
def _patch(c,te):
    r=np.arange(c,te[-1]+1);sv=(Tm[r].copy(),Sun[r].copy(),Wind[r].copy(),T24[r].copy(),T72[r].copy(),T168[r].copy())
    pr=np.arange(c-24,c);b=np.nanmean(Tm[pr]-TF[pr])
    Tm[r]=np.where(np.isfinite(TF[r]),TF[r]+b,Tm[r]);Sun[r]=np.where(np.isfinite(SF[r]),SF[r],Sun[r]);Wind[r]=np.where(np.isfinite(WF[r]),WF[r],Wind[r])
    for i in r:
        T24[i]=Tm[i-23:i+1].mean();T72[i]=Tm[i-71:i+1].mean();T168[i]=Tm[i-167:i+1].mean()
    return sv
def _restore(c,te,sv):
    r=np.arange(c,te[-1]+1);Tm[r],Sun[r],Wind[r],T24[r],T72[r],T168[r]=sv
def gbX(i,F):return np.column_stack([Tm[i],T24[i],T72[i],T168[i],Sun[i],Wind[i],R[i],F,hod[i],dow[i]])
def run(cfg,target,days,wx=False,every=7):
    """weather: None=measured; else dict with T,S,W overriding test hours is applied by the caller through globals"""
    pred={};co=None;last=-99;gb=None
    for o in days:
        c=idx_cut[o];te=np.flatnonzero(day==o)
        if o-last>=every:
            tr=np.flatnonzero((np.arange(n)<c)&ok&np.isfinite(target)&(day>=o-cfg.get("span",3000)))
            w=0.5**((c-tr)/(24*cfg.get("hl",1e6)))
            Ftr=np.where(np.isfinite(Fa[tr]),Fa[tr],flowhat(tr,c))
            if cfg.get("gb"):
                gb=GB(max_iter=cfg.get("iters",200),learning_rate=.05,max_leaf_nodes=cfg.get("leaves",15),min_samples_leaf=40,l2_regularization=1.,monotonic_cst=[-1,-1,-1,-1,0,0,0,0,0,0],random_state=0).fit(gbX(tr,Ftr),target[tr],sample_weight=w)
            else:
                co=ridge(feats(tr,cfg,Ftr),target[tr],w,cfg.get("l",1.0))
            last=o
        def p(i):
            F=flowhat(i,c)
            return np.maximum(gb.predict(gbX(i,F)) if cfg.get("gb") else feats(i,cfg,F)@co,0)
        sv=_patch(c,te) if wx else None
        x=p(te)
        if sv is not None:_restore(c,te,sv)
        k=cfg.get("lev",0)
        if k:
            rec=np.flatnonzero((np.arange(n)<c)&(day>=o-k-1)&ok&np.isfinite(target))
            if len(rec)>k*12:
                pr=p(rec);r=(target[rec].sum()+cfg.get("prior",5))/(pr.sum()+cfg.get("prior",5));x=x*np.clip(r,.5,2)**cfg.get("shr",1.)
        pred[o]=x
    return pred
def hw(days,nd,tr=False):
    out={}
    for o in days:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10)
        p=np.array([CW[m&(hod==h)].mean() if (m&(hod==h)).any() else 0 for h in range(24)]);out[o]=p[hod[te]]
    return out
def add(a,b):return {o:a[o]+b[o] for o in a}
