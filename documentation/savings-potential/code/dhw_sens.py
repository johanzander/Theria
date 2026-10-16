"""Sensitivity: hot water occupies the compressor 14 % of the hour on average (valve share, outdoor < 8 C), so heating gets 5.0 * 0.86 kWh/h."""
import pickle
from concurrent.futures import ProcessPoolExecutor
import sp_engine
import run_all as R
from sp_methods import perfect_foresight, day_ahead, summary


def job(args):
    sn, wi, cap = args
    sp_engine.COMPRESSOR_MAX_KWH = cap
    w = R.world(wi, R.SCEN[sn]); base = w.baseline(); pbar = float(w.price.mean())
    out = {"base_cost": float(base["cost"].sum())}
    for nm, f in (("perfekt", perfect_foresight), ("dygn", day_ahead)):
        _, sim = f(w)
        m = summary(w, sim, base, pbar)
        out[nm] = {k: m[k] for k in ("saving_neutral", "saving_neutral_pct")}
    return (sn, wi, cap), out


if __name__ == "__main__":
    jobs = [(sn, wi, cap) for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann") for wi in range(4) for cap in (5.0, 5.0 * 0.86)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job, jobs))
    pickle.dump(res, open(R.OUT / "dhw_sens.pkl", "wb"))
    for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann"):
        for cap in (5.0, 4.3):
            for nm in ("perfekt", "dygn"):
                s = sum(res[(sn, wi, 5.0 if cap == 5.0 else 5.0 * 0.86)][nm]["saving_neutral"] for wi in range(4))
                b = sum(res[(sn, wi, cap if cap == 5.0 else 5.0 * 0.86)]["base_cost"] for wi in range(4))
                print(f"{sn:16s} tak {cap} {nm:8s} {100*s/b:5.1f} %  {s/4:6.0f} kr/vinter")
