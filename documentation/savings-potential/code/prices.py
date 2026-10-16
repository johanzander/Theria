import json, sys, urllib.request, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
out = sys.argv[1]
days = []
d = dt.date(2022, 11, 1)
while d <= dt.date(2026, 10, 15):
    days.append(d); d += dt.timedelta(1)
def get(d):
    url = f"https://www.elprisetjustnu.se/api/v1/prices/{d:%Y}/{d:%m-%d}_SE4.json"
    for _ in range(3):
        try:
            return d, json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "curl/8.7.1"}), timeout=15))
        except Exception as e:
            err = e
    return d, None
res = {}
with ThreadPoolExecutor(12) as ex:
    for d, js in ex.map(get, days):
        if js is None: continue
        hr = defaultdict(list)
        for r in js:
            t = dt.datetime.fromisoformat(r["time_start"]).astimezone(dt.UTC).replace(minute=0, second=0)
            hr[t].append(r["SEK_per_kWh"])
        for t, v in hr.items(): res[t.isoformat()] = sum(v)/len(v)
json.dump(res, open(out + "/spot.json", "w"))
print(len(res), "hours,", len(days), "days requested")
