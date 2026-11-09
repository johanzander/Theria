from ml import *
import json
days=daylist("ho");act=actual(days);print('holdout days',len(days),date.fromordinal(days[0]),date.fromordinal(days[-1]))
Pr=pickle.load(open(d+"/price.pkl","rb"));price=np.array([Pr.get(int(x),np.nan) for x in t],float)
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
def hw7(ds):
    return hw(ds,7)
def trimmed(v):
    v=np.sort(v)
    if len(v)>=5:v=v[1:-1]
    elif len(v)>=3:v=v[:-1]
    return v.mean() if len(v) else np.nan
def base(ds,wx):
    P={k:{} for k in("flat48","prof7","reg28+hw7","planmodel+hw7")};h7=hw(ds,7)
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);h=hod[te];sv=None
        import ml
        if wx:sv=ml._patch(c,te)
        T=Tm[te].copy()
        if sv is not None:ml._restore(c,te,sv)
        x=np.arange(n)<c;i48=np.flatnonzero(x&ok)[-48:];P["flat48"][o]=np.full(len(te),E[i48].mean())
        m=x&ok&(day>=o-8);p=np.array([trimmed(E[m&(hod==k)]) if (m&(hod==k)).any() else 0 for k in range(24)]);P["prof7"][o]=p[h]
        tr=x&ok&(day>=o-29);po=np.polyfit(Tm[tr],CH[tr],1);P["reg28+hw7"][o]=np.maximum(np.polyval(po,T),0)+h7[o]
        P["planmodel+hw7"][o]=np.maximum(-1.0264-.2362*T+.259*R[te],0)+h7[o]
    return P
def costerr(P,ds):
    de=[]
    for o in ds:
        pr=price[day==o]
        if np.isfinite(pr).all():de.append((P[o]*pr).sum()-(act[o]*pr).sum())
    return np.abs(de).mean(),np.mean(de),len(de)
B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365};w=hwp(days,28);avg=lambda a,b:{o:(a[o]+b[o])/2 for o in a}
OUT={}
for wx in (False,True):
    print("=== weather:", "forecast, about a day ahead" if wx else "measured")
    L1=run({**B,"lev":3,"shr":.5},CH,days,wx=wx);G1=run({"gb":1,"hl":365,"leaves":15,"iters":150,"lev":3,"shr":.5},CH,days,wx=wx,every=14)
    F=add(avg(L1,G1),w);S=add(L1,w);P=base(days,wx);P["FINAL mean(linear,gb)+hw28wd"]=F;P["SIMPLE linear+hw28wd"]=S
    for k in ("FINAL mean(linear,gb)+hw28wd","SIMPLE linear+hw28wd","reg28+hw7","planmodel+hw7","flat48","prof7"):
        s=score(P[k],act,days);c=costerr(P[k],days);show(k,s);print(f"     cost error {c[0]:.2f} kr/day, bias {c[1]:+.2f}, n {c[2]}")
    OUT[wx]=P
    # paired bootstrap of day MAE: FINAL against reg28+hw7 and against SIMPLE
    a=np.array([abs(F[o].sum()-act[o].sum()) for o in days]);rng=np.random.default_rng(0)
    for nm in ("reg28+hw7","SIMPLE linear+hw28wd"):
        b=np.array([abs(P[nm][o].sum()-act[o].sum()) for o in days]);dd=[(a-b)[rng.integers(0,len(a),len(a))].mean() for _ in range(4000)];print(f"     FINAL minus {nm}: {np.mean(a-b):+.2f} kWh/day, 95% [{np.percentile(dd,2.5):+.2f},{np.percentile(dd,97.5):+.2f}]")
    if not wx:
        for nm,sel in (("<8C",lambda o:Tday[o]<8),(">=8C",lambda o:Tday[o]>=8)):
            ds=[o for o in days if sel(o)];print(nm,len(ds));show("  FINAL",score(F,act,ds));show("  reg28",score(P["reg28+hw7"],act,ds))
pickle.dump(OUT,open(d+"/h1.pkl","wb"))
# flow estimate error
e=[]
for o in days:
    c=idx_cut[o];te=np.flatnonzero(day==o);k=np.isfinite(Fa[te])
    if k.any():e+=list(flowhat(te[k],c)-Fa[te[k]])
e=np.array(e);print("flow target estimated a day ahead, measured weather: MAE",np.abs(e).mean().round(2),"bias",e.mean().round(2),"n",len(e))
