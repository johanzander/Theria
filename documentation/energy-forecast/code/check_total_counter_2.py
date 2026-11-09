import sys;sys.path.insert(0,sys.argv[1])
from ml import *
T=pickle.load(open(d+"/hs/tot.pkl","rb"));tot=T["tot"]
days=daylist("ho");dd=[];
for o in days:
    m=day==o
    if np.isfinite(tot[m]).all():dd.append((E[m].sum(),tot[m].sum()))
x=np.array(dd);diff=x[:,0]-x[:,1];print("holdout days",len(x),"CH+CW",x[:,0].sum().round(0),"total counter",x[:,1].sum().round(0),"mean diff",diff.mean().round(3),"MAD",np.abs(diff).mean().round(3),">1 kWh:",(np.abs(diff)>1).sum())
dv=daylist("dev");act=actual(dv)
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);mm=m&((dow>=5)==(dow[te[0]]>=5));q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
w=hwp(dv,28)
show("SIMPLE (flow) dev",score(add(run({"lag":1,"flow":1,"sun":1,"wind":1,"hl":365,"lev":3,"shr":.5},CH,dv),w),act,dv))
show("setpoint, no flow, dev",score(add(run({"lag":1,"rts":1,"sun":1,"wind":1,"hl":365,"lev":3,"shr":.5},CH,dv),w),act,dv))
