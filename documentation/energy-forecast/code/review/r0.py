from ml import *
import numpy as np
print(n, date.fromordinal(day[0]), date.fromordinal(day[-1]))
for k,v in D.items():
    if k!='t': print(k, round(float(np.isfinite(v).mean()),3), np.round(np.nanpercentile(v,[0,1,50,99,100]),2))
print({k:(len(v[0]),list(v[1].keys())) for k,v in W.items()})
for nm,days in (('dev',daylist('dev')),('ho',daylist('ho'))): print(nm,len(days))
