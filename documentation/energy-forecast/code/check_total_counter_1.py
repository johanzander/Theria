import sys,os,pickle,numpy as np
S=sys.argv[1];sys.path.insert(0,S)
src=open(S+"/d1.py").read()
from core.theria.influxdb_helper import query_influx as Q
pre=src.split("D=dict(")[0]
g={"__file__":S+"/d1.py"};exec(pre,g)
e=g["counter"]("sensor.compr_consump_tot",14);grid=g["grid"]
pickle.dump(dict(t=grid[:-1],tot=e),open(S+"/hs/tot.pkl","wb"))
