from ml import *
import ml,numpy as np,pickle
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
def reg(ds,span=28,withR=False,hl=None):
    P={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o)
        tr=(np.arange(n)<c)&ok&np.isfinite(CH)&(day>=o-span-1)
        if withR:
            X=lambda i:np.column_stack([np.ones(len(i)),np.maximum(16-Tm[i],0),R[i]-18]);tri=np.flatnonzero(tr);co=np.linalg.lstsq(X(tri),CH[tri],rcond=None)[0];P[o]=np.maximum(X(te)@co,0)
        else:
            po=np.polyfit(Tm[tr],CH[tr],1);P[o]=np.maximum(np.polyval(po,Tm[te]),0)
    return P
def daily_total(ds,lev=True):
    """Fit daily heating kWh on daily features (hl 365), then rescale the SIMPLE hourly shape to that total."""
    P={};co=None;last=-99
    def fX(o):
        i=np.flatnonzero(day==o);T=Tm[i];h=np.maximum(16-T,0)
        return np.array([1,h.mean(),np.maximum(10-T,0).mean(),np.maximum(4-T,0).mean(),np.maximum(-2-T,0).mean(),np.maximum(16-T24[i],0).mean(),np.maximum(16-T72[i],0).mean(),R[i].mean()-18,(R[i]-18).mean()*h.mean()/10,Sun[i].mean()/100,Wind[i].mean()/10*h.mean()/10])
    return fX
for nm in ('dev',):
    ds=daylist(nm);act=actual(ds);h=hwp(ds,28);h7=hw(ds,7)
    B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365}
    L1=run({**B,"lev":3,"shr":.5},CH,ds);G1=run({"gb":1,"hl":365,"leaves":15,"iters":150,"lev":3,"shr":.5},CH,ds,every=14)
    avg=lambda *a:{o:sum(x[o] for x in a)/len(a) for o in a[0]}
    F=add(avg(L1,G1),h);S=add(L1,h);show('FINAL dev',score(F,act,ds));show('SIMPLE dev',score(S,act,ds))
    r28=reg(ds);show('reg28+hw7',score(add(r28,h7),act,ds));show('reg28+hw28wd',score(add(r28,h),act,ds))
    r28R=reg(ds,28,True);show('reg28 with R +hw28wd',score(add(r28R,h),act,ds))
    r14R=reg(ds,14,True);show('reg14 with R +hw28wd',score(add(r14R,h),act,ds))
    r56R=reg(ds,56,True);show('reg56 with R +hw28wd',score(add(r56R,h),act,ds))
    show('avg SIMPLE, reg28R',score(add(avg(L1,r28R),h),act,ds))
    show('avg FINAL, reg28R',score(add(avg(L1,G1,r28R),h),act,ds))
    show('avg FINAL, reg28 (no R)',score(add(avg(L1,G1,r28),h),act,ds))
    # Bias: per-cold
    X=[]
    # R interact: rts variant
    Rc=run({**B,"lev":3,"shr":.5,"rts":1},CH,ds);show('SIMPLE+rts',score(add(Rc,h),act,ds))
    NoF=run({"lag":1,"sun":1,"wind":1,"hl":365,"lev":3,"shr":.5,"rts":1},CH,ds);show('SIMPLE with R but no flow',score(add(NoF,h),act,ds))
    show('L1 vs reg28R mean, 3-way with GB',score(add(avg(L1,G1,r28R),h),act,ds))
    # daily model
    fX=daily_total(ds);P={};co=None;last=-99
    train_days=sorted(set(day))
    for o in ds:
        c=idx_cut[o]
        if o-last>=7:
            t_=[x for x in train_days if x<o-1 and (day==x).sum()>=23 and ok[day==x].all() and np.isfinite(CH[day==x]).all()];XX=np.array([fX(x) for x in t_]);yy=np.array([CH[day==x].sum() for x in t_]);w=0.5**((o-np.array(t_))/365);co=ridge(XX,yy,w,1.0);last=o
        tot=max(float(fX(o)@co),0);p=L1[o];P[o]=p*tot/max(p.sum(),1e-3)
    show('daily-total ridge, SIMPLE shape',score(add(P,h),act,ds));
    show('avg SIMPLE/daily',score(add(avg(L1,P),h),act,ds))
    pickle.dump(dict(L1=L1,G1=G1,r28R=r28R,h=h,S=S,F=F),open(d+'/review/dev_runs.pkl','wb'))
