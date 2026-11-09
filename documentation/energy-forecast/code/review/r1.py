from ml import *
import collections
# coverage per season-half
seas=lambda o:date.fromordinal(o)
ym=np.array([seas(x).year*100+seas(x).month for x in day])
print("month  Ffin Rfin Ofin  CHfin CWfin qh  bad  Rmean Rstd meanT dailyE dayN")
for m in sorted(set(ym)):
    i=ym==m
    print(m, *[round(float(np.isfinite(x[i]).mean()),2) for x in (D['F'],D['R'],D['O'],D['ch'],D['cw'],D['qh'])], round(float(bad[i].mean()),2), round(float(np.nanmean(D['R'][i])) if np.isfinite(D['R'][i]).any() else float('nan'),2), round(float(np.nanstd(D['R'][i])) if np.isfinite(D['R'][i]).any() else float('nan'),2), round(float(Tm[i].mean()),1), round(float(np.nansum(E[i])/max(1,(~np.isnan(E[i])).sum())*24),1), len(set(day[i])))
print("first F",datetime.fromtimestamp(t[np.isfinite(D['F'])][0],L),"first R",datetime.fromtimestamp(t[np.isfinite(D['R'])][0],L))
print('Tm nan',np.isnan(Tm).sum(),'Sun nan',np.isnan(Sun).sum(),'Wind nan',np.isnan(Wind).sum(),'TF finite',np.isfinite(TF).mean())
