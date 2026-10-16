"""Independent check of the LP: backward dynamic programming (the bess-manager approach) on short windows."""
import pickle, numpy as np
from scipy.interpolate import RegularGridInterpolator
from sp_engine import *

def dp_window(w: World, lam: float, r_step=0.02, s_step=0.25, u_step=0.25):
    c, p = w.cfg, w.pump
    n = w.n
    Rg = np.arange(c.band_lo, c.band_hi + 1e-9, r_step)
    s_lo = min(w.s_ref.min(), (w._flow_i(c.u_lo, 0))) - 3; s_hi = max(w.s_ref.max(), c.flow_cap if c.flow_cap < 1e6 else 50) + 1
    Sg = np.arange(np.floor(s_lo), s_hi + 1e-9, s_step)
    U = np.arange(c.u_lo, c.u_hi + 1e-9, u_step)
    RR, SS = np.meshgrid(Rg, Sg, indexing="ij")
    V = np.where(RR >= c.target - 1e-9, 0.0, 1e6)  # end at or above reference
    pol = np.zeros((n, *RR.shape), dtype=np.int16)
    for t in range(n - 1, -1, -1):
        interp = RegularGridInterpolator((Rg, Sg), V, bounds_error=False, fill_value=1e6)
        best = np.full(RR.shape, np.inf); arg = np.zeros(RR.shape, dtype=np.int16)
        for k, u in enumerate(U):
            u_eff = min(u, w.u_max[t]); u_eff = max(u_eff, w.u_min[t])
            comp, aux = w.energy(np.full(n, u_eff))
            e = comp[t] + aux[t]
            flow = min(w._flow_i(u_eff, t), c.flow_cap)
            R2 = RR + w.ks * (SS - RR) - w.ko * (RR - w.t[t]) + w.drift[t]
            S2 = SS + w.a * (flow - SS)
            ok = (R2 >= c.band_lo) & (R2 <= c.band_hi)
            val = w.price[t] * e - lam * (R2 - c.target) + interp(np.stack([np.clip(R2, Rg[0], Rg[-1]), np.clip(S2, Sg[0], Sg[-1])], -1).reshape(-1, 2)).reshape(RR.shape)
            val = np.where(ok, val, 1e6)
            better = val < best
            best = np.where(better, val, best); arg = np.where(better, k, arg)
        V = best; pol[t] = arg
    # roll forward from reference state
    r, s = c.target, w.s_ref[0]
    interpP = None
    u_path = np.zeros(n); total = 0.0
    for t in range(n):
        ri = np.argmin(abs(Rg - r)); si = np.argmin(abs(Sg - s))
        u = U[pol[t][ri, si]]; u_path[t] = min(max(u, w.u_min[t]), w.u_max[t])
        r, s = w.step(t, r, s, u_path[t])
    sim = w.simulate(u_path)
    obj = float(sim["cost"].sum() - lam * np.sum(sim["room"] - c.target))
    return obj, u_path, sim

out = []
hours, ser = load_study()
for wi, start_day in ((3, 40), (3, 90), (2, 60), (1, 20), (0, 70)):
    lab, a, b = winters()[wi]
    h, s = window(hours, ser, a, b)
    i0 = start_day * 24; i1 = i0 + 120
    for sname, cfg in (("Idag", Cfg(neutral=False)), ("Utvidgat", Cfg(band_lo=16, band_hi=23, u_lo=8, u_hi=20, flow_cap=40, neutral=False))):
        w = World(h[i0:i1], s["price"][i0:i1], s["t_out"][i0:i1], s["sun"][i0:i1], cfg)
        lam = 0.26 * float(np.mean(w.price))
        base = w.baseline()
        base_obj = float(base["cost"].sum())
        u_lp, res = optimise(w, comfort_weight=lam, neutral=False)
        sim_lp = w.simulate(u_lp)
        lp_obj = float(sim_lp["cost"].sum() - lam * np.sum(sim_lp["room"] - cfg.target))
        dp_obj, u_dp, sim_dp = dp_window(w, lam)
        out.append(dict(winter=lab, day=start_day, scen=sname, base=base_obj, lp=lp_obj, dp=dp_obj))
        print(lab, start_day, sname, "base %.2f  LP %.3f  DP %.3f  (LP saves %.2f, DP saves %.2f; gap %.2f%% of saving)" % (base_obj, lp_obj, dp_obj, base_obj-lp_obj, base_obj-dp_obj, 100*(dp_obj-lp_obj)/max(base_obj-lp_obj,1e-9)), flush=True)
pickle.dump(out, open("results/dpcheck.pkl", "wb"))
