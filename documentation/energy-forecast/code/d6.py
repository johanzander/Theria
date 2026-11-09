import numpy as np,pickle,os
from datetime import datetime,timezone
from core.theria.influxdb_helper import query_influx as Q
d=os.path.dirname(os.path.abspath(__file__));A=int(datetime(2022,10,1,tzinfo=timezone.utc).timestamp())
r=Q([f'SELECT mean("value") FROM "sensor.nord_pool_se4_current_price" WHERE time>={A}s GROUP BY time(1h) fill(none)'])[0]
pickle.dump({x[0]//1000:x[1] for x in r},open(d+"/price.pkl","wb"));print(len(r))
