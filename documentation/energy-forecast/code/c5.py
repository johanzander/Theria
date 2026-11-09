from ml import *
days=daylist("dev");act=actual(days)
def hwp(ds,nd):
    out={}
    for o in ds:
        c=idx_cut[o];te=np.flatnonzero(day==o);m=(np.arange(n)<c)&(day>=o-nd-1)&np.isfinite(CW)&(CW>=0)&(CW<10);we=dow[te[0]]>=5;mm=m&((dow>=5)==we)
        q=np.array([CW[mm&(hod==h)].mean() if (mm&(hod==h)).any() else 0 for h in range(24)]);out[o]=q[hod[te]]
    return out
w=hwp(days,28);B={"lag":1,"flow":1,"sun":1,"wind":1,"hl":365}
L0=run(B,CH,days);L1=run({**B,"lev":3,"shr":.5},CH,days)
G0=run({"gb":1,"hl":365,"leaves":15,"iters":150},CH,days,every=14);G1=run({"gb":1,"hl":365,"leaves":15,"iters":150,"lev":3,"shr":.5},CH,days,every=14)
avg=lambda a,b:{o:(a[o]+b[o])/2 for o in a}
for k,v in (("linear",L0),("linear lev3 .5",L1),("gb",G0),("gb lev3 .5",G1),("mean linear+gb",avg(L0,G0)),("mean lev3 .5",avg(L1,G1)),("mean linear lev + gb",avg(L1,G0))):show(k,score(add(v,w),act,days))
