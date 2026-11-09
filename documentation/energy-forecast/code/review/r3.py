import pickle,numpy as np
from datetime import datetime,timezone
D=pickle.load(open('d1.pkl','rb'))
for k,v in D.items():
    a=np.asarray(v) if not isinstance(v,dict) else None
    print(k,type(v).__name__, None if a is None else a.shape, None if a is None else a.flat[:3], )
R=pickle.load(open('rev/data.pkl','rb'));print(type(R), list(R.keys())[:40] if isinstance(R,dict) else len(R))
