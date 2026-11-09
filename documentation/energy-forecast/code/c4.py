from ml import *
import ml
days=daylist("dev");act=actual(days);w=hw(days,7);res=pickle.load(open(d+"/c3.pkl","rb"))
Tg=np.array([-15,-7,2,7,12.]);Fg=np.array([35,45,55.]);G=np.array([[2.33,2.01,1.60],[2.69,2.32,1.85],[4.03,3.20,2.48],[4.72,3.61,2.58],[5.68,3.99,2.70]])
def cop(T,F):
    T=np.clip(T,-15,12);F=np.clip(F,35,55);i=np.clip(np.searchsorted(Tg,T)-1,0,3);j=np.clip(np.searchsorted(Fg,F)-1,0,1)
    a=(T-Tg[i])/(Tg[i+1]-Tg[i]);b=(F-Fg[j])/(Fg[j+1]-Fg[j])
    return G[i,j]*(1-a)*(1-b)+G[i+1,j]*a*(1-b)+G[i,j+1]*(1-a)*b+G[i+1,j+1]*a*b
def runx(fn,days,target=CH,hl=365,every=7,l=1.0):
    pred={};co=None;last=-99
    for o in days:
        c=idx_cut[o];te=np.flatnonzero(day==o)
        if o-last>=every:
            tr=np.flatnonzero((np.arange(n)<c)&ok&np.isfinite(target));wt=0.5**((c-tr)/(24*hl));Ft=np.where(np.isfinite(Fa[tr]),Fa[tr],flowhat(tr,c))
            co=ridge(fn(tr,Ft),target[tr],wt,l);last=o
        pred[o]=np.maximum(fn(te,flowhat(te,c))@co,0)
    return pred
def f_only(i,F):return np.column_stack([np.ones(len(i)),F,F*F])
def f_only_on(i,F):return np.column_stack([np.ones(len(i)),np.maximum(F-28,0),np.maximum(F-34,0),np.maximum(F-40,0)])
def Q(F):return np.maximum(F-20,1)**1.2
def ph1(i,F):return np.column_stack([np.ones(len(i)),Q(F)/cop(Tm[i],F)])
def ph2(i,F):return np.column_stack([np.ones(len(i)),np.maximum(16-Tm[i],0)/cop(Tm[i],F)])
def ph3(i,F):
    C=cop(Tm[i],F);return np.column_stack([np.ones(len(i)),np.maximum(16-Tm[i],0)/C,Q(F)/C])
def ph4(i,F):
    C=cop(Tm[i],F);h=np.maximum(16-Tm[i],0)
    return np.column_stack([np.ones(len(i)),h/C,Q(F)/C,np.maximum(16-T24[i],0)/C,np.maximum(16-T72[i],0)/C,Sun[i]/100/C,Wind[i]/10*h/10/C])
for k,f in (("flow only (quadratic)",f_only),("flow only (hinges)",f_only_on),("physics: emitter(F)/COP",ph1),("physics: degree-hours/COP(T,F)",ph2),("physics: both /COP",ph3),("physics: both + lags, sun, wind /COP",ph4)):
    res[k]=runx(f,days);show(k,score(add(res[k],w),act,days))
pickle.dump(res,open(d+"/c4.pkl","wb"))
# hot water candidates, with the best heating so far
a=res["linear hl=365"];g=res["gb leaves15 it150 hl365"];H={o:(a[o]+g[o])/2 for o in a}
def hwp(nd,dowsplit=False):
    out={}
    for o in days:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10)
        p=np.array([CW[m&(hod==h)].mean() if (m&(hod==h)).any() else 0 for h in range(24)])
        if dowsplit:
            we=dow[te[0]]>=5;mm=m&((dow>=5)==we);q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);p=q
        out[o]=p[hod[te]]
    return out
for nd in (3,7,14,28,56,112):show(f"hot water {nd} d",score(add(H,hwp(nd)),act,days))
for nd in (14,28,56):show(f"hot water {nd} d, weekday/weekend",score(add(H,hwp(nd,True)),act,days))
gbw=run({"gb":1,"hl":365,"leaves":15,"iters":150},CW,days,every=14);show("hot water gb",score(add(H,gbw),act,days))
