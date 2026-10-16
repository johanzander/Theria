"""Practical (day-ahead, Nordpool-timed) savings for realistic limits: flow <= 40 C, room ceiling 22, ramp limits, setpoint range."""
import pickle, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import run_all as R
from sp_engine import Cfg
from sp_methods import day_ahead, summary


def cfg_of(lo, hi, up, down, band=(17.0, 22.0)):
    return Cfg(band_lo=band[0], band_hi=band[1], u_lo=lo, u_hi=hi, flow_cap=40.0, max_step=up, max_step_down=down)


def job(args):
    wi, lo, hi, up, down, term, hour, *rest = args
    w = R.world(wi, cfg_of(lo, hi, up, down, *rest)); base = w.baseline(); pbar = float(w.price.mean())
    _, sim = day_ahead(w, terminal_mult=term, plan_hour=hour)
    m = summary(w, sim, base, pbar)
    return args, {"saving": m["saving_neutral"], "base": m["base_cost"], "min_room": m["min_room"], "max_room": m["max_room"]}


def pooled(res, **sel):
    keys = [k for k in res if all(k[i] == v for i, v in enumerate(sel.values()) if v is not None)]
    return keys


if __name__ == "__main__":
    which = sys.argv[1]
    jobs = []
    if which == "range":
        jobs = [(wi, lo, hi, 2.0, 2.0, 1.0, 14) for wi in range(4) for lo in (8, 10, 12, 14, 16) for hi in (18, 19, 20, 22, 24)]
    elif which == "ramp":
        jobs = [(wi, 8, 20, up, dn, 1.0, 14) for wi in range(4) for up, dn in ((None, None), (2.0, None), (2.0, 2.0), (1.0, 1.0), (4.0, 4.0))]
    elif which == "term":
        lo, hi = float(sys.argv[2]), float(sys.argv[3])
        jobs = [(wi, lo, hi, 2.0, 2.0, t, 14) for wi in range(4) for t in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)]
    elif which == "band":
        lo, hi = float(sys.argv[2]), float(sys.argv[3])
        jobs = [(wi, lo, hi, 2.0, 2.0, 1.0, 14, (bl, 22.0)) for wi in range(4) for bl in (16.0, 17.0, 18.0, 18.5, 19.0)]
    elif which == "hour":
        lo, hi = float(sys.argv[2]), float(sys.argv[3])
        jobs = [(wi, lo, hi, 2.0, 2.0, 1.0, h) for wi in range(4) for h in (12, 13, 14, 15, 16, 18)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job, jobs))
    pickle.dump(res, open(R.OUT / f"sweep_{which}.pkl", "wb"))
    groups = {}
    for k, v in res.items():
        groups.setdefault(k[1:], []).append((k[0], v))
    for g, items in sorted(groups.items(), key=lambda x: str(x[0])):
        s = sum(v["saving"] for _, v in items); b = sum(v["base"] for _, v in items)
        per = [100 * v["saving"] / v["base"] for _, v in sorted(items, key=lambda x: x[0])]
        print(g, f"{100*s/b:5.1f} %  {s/4:5.0f} kr/vinter  per vinter {[round(x,1) for x in per]}  min rum {min(v['min_room'] for _, v in items):.1f}")
