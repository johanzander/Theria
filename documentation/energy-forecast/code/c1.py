from bm import *
days=daylist("dev");act=actual(days);print('dev days',len(days))
def trimmed(v):
    v=np.sort(v)
    if len(v)>=5:v=v[1:-1]
    elif len(v)>=3:v=v[:-1]
    return v.mean() if len(v) else np.nan
def prof(x,o,cut,nd,tr=True):
    m=(np.arange(n)<cut)&(day>=o-nd-1)&np.isfinite(x)&(x>=0)&(x<10)
    return np.array([(trimmed(x[m&(hod==h)]) if tr else x[m&(hod==h)].mean()) if (m&(hod==h)).any() else 0 for h in range(24)])
# baselines
P={k:{} for k in("flat48","prof7","reg28+hw7","planmodel+hw7","hingeall+hw7")}
ok=~bad
for o in days:
    c=idx_cut[o];te=np.flatnonzero(day==o);h=hod[te];T=Tm[te];Rt=R[te]
    x=np.arange(n)<c;i48=np.flatnonzero(x&ok)[-48:];P["flat48"][o]=np.full(len(te),E[i48].mean())
    P["prof7"][o]=prof(E,o,c,7)[h];hw7=prof(CW,o,c,7,False)[h]
    tr=x&ok&(day>=o-29);po=np.polyfit(Tm[tr],CH[tr],1);P["reg28+hw7"][o]=np.maximum(np.polyval(po,T),0)+hw7
    P["planmodel+hw7"][o]=np.maximum(-1.0264-.2362*T+.259*Rt,0)+hw7
    tr=x&ok;X=lambda v:np.column_stack([np.ones(len(v)),np.maximum(16-v,0)]);co=np.linalg.lstsq(X(Tm[tr]),CH[tr],rcond=None)[0];P["hingeall+hw7"][o]=np.maximum(X(T)@co,0)+hw7
for k,v in P.items():show(k,score(v,act,days))
