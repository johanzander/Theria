import numpy as np
from ball import *
if __name__ == "__main__":
    rows = []
    for wi in range(4):
        w = world(wi, cfg); day = local_days(w); p = w.price
        for d in np.unique(day)[3:-3]:
            idx = np.where(day == d)[0]
            if len(idx) != 24: continue
            s = np.sort(p[idx]); rows.append((s[-6:].mean() - s[:6].mean(), wi, d, idx[0], float(w.t[idx].mean())))
    rows.sort(reverse=True)
    pick = []
    for lo, hi in ((-2, 4), (-12, -4)):   # mild-ish and cold day, among the 40 biggest spreads
        c = [r for r in rows[:150] if lo <= r[4] <= hi]
        pick.append(c[0])
    for spread, wi, d, j0, tm in pick:
        w0 = world(wi, cfg); w = w0.slice(j0 - 48, j0 + 24 + 72); base = w.baseline()
        day = local_days(w); mid = day == np.unique(day)[2]; idx = np.where(mid)[0]; p = w.price
        o = idx[np.argsort(p[idx], kind="stable")]
        for off in (-3.0, -8.0):
            du = np.zeros(w.n); du[o[:6]] += 8.0; du[o[-6:]] += off
            u, sim = run_policy(w, du)
            pw = float(np.percentile(w.price, 90))
            cold_end = max(0.0, w.cfg.target - sim["room"][-1])
            settle_c = pw * w.kwh_per_c_stock() * cold_end
            saving = base["cost"].sum() - sim["cost"].sum() - settle_c
            print(f"dag spridning topp6-botten6 {spread:.2f} kr, medeltemp ute {tm:.1f}C, ned {off:+.0f}: spar {saving:.1f} kr av {base['cost'][idx].sum():.0f} kr/dygn "
                  f"({100*saving/base['cost'][idx].sum():.0f}%), kWh dygnet {base['kwh'][idx].sum():.0f}->{sim['kwh'][idx].sum():.0f}, "
                  f"rum min {sim['room'].min():.1f} slut {sim['room'][-1]:.2f}, tillägg slut {settle_c:.1f} kr")
