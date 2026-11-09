from ml import *
import pickle
OUT=pickle.load(open(d+'/h1.pkl','rb'));days=daylist('ho');act=actual(days)
for wx in (False,True):
    P=OUT[wx]; print('wx',wx)
    for k in P: show(k,score(P[k],act,days))
    F=P['FINAL mean(linear,gb)+hw28wd']
    e=lambda p:np.array([p[o].sum()-act[o].sum() for o in days])
    a=np.abs(e(F));rng=np.random.default_rng(1)
    for nm in ('reg28+hw7','SIMPLE linear+hw28wd'):
        b=np.abs(e(P[nm]));dd=a-b;m=len(dd);res={}
        # iid and moving-block by day index within contiguous test day list
        for bl in (1,7,14,21,28):
            st=[]
            for _ in range(5000):
                idx=[];
                while len(idx)<m:
                    s=rng.integers(0,m);idx+=[(s+j)%m for j in range(bl)]
                st.append(dd[np.array(idx[:m])].mean())
            res[bl]=(np.percentile(st,2.5),np.percentile(st,97.5))
        ac=[np.corrcoef(dd[:-k],dd[k:])[0,1] for k in (1,2,3,7)]
        print(' ',nm,'mean',round(dd.mean(),2),'autocorr lag1,2,3,7',np.round(ac,2),{k:(round(v[0],2),round(v[1],2)) for k,v in res.items()})
    # bias CI with block bootstrap
    b=e(F);rng=np.random.default_rng(2);st=[]
    for _ in range(5000):
        idx=[];
        while len(idx)<m:
            s=rng.integers(0,m);idx+=[(s+j)%m for j in range(14)]
        st.append(b[np.array(idx[:m])].mean())
    print('  FINAL bias',b.mean().round(2),'block14 95%',np.round(np.percentile(st,[2.5,97.5]),2),' autocorr e lag1',np.round(np.corrcoef(b[:-1],b[1:])[0,1],2))
    # pct of total
    tot=sum(act[o].sum() for o in days);print('  mean day kWh',round(tot/len(days),1),'bias% ',round(100*b.sum()/tot,2))
    # by season
    for nm,sel in (('oct25-apr26',lambda o:o<date(2026,6,1).toordinal()),('oct-nov26',lambda o:o>=date(2026,6,1).toordinal())):
        ds=[o for o in days if sel(o)]
        for k in ('FINAL mean(linear,gb)+hw28wd','reg28+hw7'):
            show(f'  {nm} {k[:14]}',score(P[k],act,ds))
