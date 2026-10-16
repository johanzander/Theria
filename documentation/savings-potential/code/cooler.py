import numpy as np
from sp_engine import Cfg
from sp_methods import run_policy
import run_all as R
if __name__ == "__main__":
    for tgt in (19.0, 18.5, 18.0):
        tot_s = tot_b = tot_k = tot_bk = 0.0; rows = []
        for wi in range(4):
            w = R.world(wi, Cfg(band_lo=16, band_hi=23)); base = w.baseline()
            lo, hi = -6.0, 0.0
            for _ in range(24):
                mid = 0.5 * (lo + hi); _, sim = run_policy(w, np.zeros(w.n), mid)
                if sim["room"].mean() < tgt: lo = mid
                else: hi = mid
            _, sim = run_policy(w, np.zeros(w.n), mid)
            s = base["cost"].sum() - sim["cost"].sum(); k = base["kwh"].sum() - sim["kwh"].sum()
            rows.append((s, k, 100 * s / base["cost"].sum(), mid, sim["room"].mean()))
            tot_s += s; tot_b += base["cost"].sum(); tot_k += k; tot_bk += base["kwh"].sum()
        print(f"komfort {tgt}: spar {tot_s/4:.0f} kr/vinter ({100*tot_s/tot_b:.1f} %), {tot_k/4:.0f} kWh/vinter ({100*tot_k/tot_bk:.1f} %), börvärde {np.mean([r[3] for r in rows]):+.2f} °C; per vinter kr {[round(r[0]) for r in rows]}")
