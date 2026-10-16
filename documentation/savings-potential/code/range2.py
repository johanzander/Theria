import sweep
from concurrent.futures import ProcessPoolExecutor
if __name__ == "__main__":
    jobs = [(wi, lo, 22.0, 2.0, 2.0, 1.0, 14, (18.5, 22.0)) for wi in range(4) for lo in (13.0, 14.0, 15.0, 15.5, 16.0)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(sweep.job, jobs))
    g = {}
    for k, v in res.items(): g.setdefault(k[1], []).append(v)
    for lo, items in sorted(g.items()):
        s = sum(v["saving"] for v in items); b = sum(v["base"] for v in items)
        print(f"rts {lo:.0f}–22 (golv 18,5): {100*s/b:.1f} %  {s/4:.0f} kr/vinter  min rum {min(v['min_room'] for v in items):.1f}")
