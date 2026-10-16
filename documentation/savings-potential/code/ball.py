import numpy as np
from sp_engine import *
from sp_methods import *
HOURS, SER = load_study(); WIN = winters()
def world(wi, cfg):
    lab, a, b = WIN[wi]; h, s = window(HOURS, SER, a, b)
    return World(h, s["price"], s["t_out"], s["sun"], cfg)
cfg = Cfg(band_lo=16, band_hi=23, u_lo=8, u_hi=20, flow_cap=40)
if __name__ == "__main__":
    tot = {}
    for wi in range(4):
        w = world(wi, cfg); base = w.baseline(); pbar = float(w.price.mean())
        day = local_days(w); p = w.price
        top = np.mean([np.sort(p[day==d])[-6:].mean() for d in np.unique(day)])
        low = np.mean([np.sort(p[day==d])[:6].mean() for d in np.unique(day)])
        row = [f"{WIN[wi][0]} spot mean {pbar:.2f} top6 {top:.2f} low6 {low:.2f} base {base['cost'].sum():.0f}"]
        for (n, a) in ((6, 2.0), (6, 4.0), (6, 8.0), (4, 8.0)):
            # shut-off depth a (down) with matching preheat up
            du = np.zeros(w.n)
            for d in np.unique(day):
                idx = np.where(day==d)[0]; o = idx[np.argsort(p[idx], kind="stable")]
                du[o[:n]] += a; du[o[-n:]] -= 8.0
            _, sim, _ = run_neutral(w, du)
            m = summary(w, sim, base, pbar)
            row.append(f"off6/pre{n}+{a}: {m['saving_neutral_pct']:.1f}% ({m['saving_neutral']:.0f}) room {m['min_room']:.1f}-{m['max_room']:.1f}")
        _, sim = perfect_foresight(w); m = summary(w, sim, base, pbar)
        row.append(f"LP ceiling {m['saving_neutral_pct']:.1f}% ({m['saving_neutral']:.0f})")
        print("\n  ".join(row), flush=True)
