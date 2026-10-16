"""Run every experiment and cache the results (pickle). Usage: python run_all.py [experiment ...]"""
import pickle, sys, time, itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from sp_engine import *
from sp_methods import *

OUT = SCRATCH / "results"
OUT.mkdir(exist_ok=True)
HOURS, SER = load_study()
WIN = winters()

SCEN = {
    "S1 Idag": Cfg(),
    "S2 Brett band": Cfg(band_lo=16, band_hi=23),
    "S3 Nedsänkning": Cfg(band_lo=16, band_hi=23, u_lo=8, u_hi=20, flow_cap=40),
    "S4 Stort spann": Cfg(band_lo=16, band_hi=23, u_lo=8, u_hi=26, flow_cap=40),
}


def world(wi, cfg, price=None):
    lab, a, b = WIN[wi]
    h, s = window(HOURS, SER, a, b)
    return World(h, s["price"] if price is None else price, s["t_out"], s["sun"], cfg)


def daily(w, arr):
    day = local_days(w)
    return np.array([arr[day == d].sum() for d in np.unique(day)])


# ---------- A. headline: every method x scenario x winter --------------------------------------
RULE_N = [(n, a) for n in (2, 4, 6, 8, 12) for a in (0.5, 1, 2, 3, 4, 6)]
RULE_K = [(k) for k in (0.25, 0.5, 1, 1.5, 2, 3, 4)]
RULE_PAIR = [(a) for a in (0.5, 1, 2, 3, 4, 6)]
RULE_MD = [(k) for k in (0.25, 0.5, 1, 1.5, 2, 3, 4)]


def job_methods(args):
    sname, wi = args
    cfg = SCEN[sname]
    w = world(wi, cfg)
    base = w.baseline()
    pbar = float(w.price.mean())
    tau_h = 1.0 / (w.ks + w.ko)
    out = {"base_daily": daily(w, base["cost"]), "n": w.n}
    def rec(key, sim):
        m = summary(w, sim, base, pbar)
        m["daily_saving"] = daily(w, base["cost"]) - daily(w, sim["cost"])
        m["settle"] = m["settled"] - m["cost"]
        out[key] = m
        return sim
    _, sim = perfect_foresight(w); rec("LP perfekt", sim)
    for ahead, nm in ((1, "LP dygnet före"), (2, "LP 2 dygn"), (3, "LP 3 dygn"), (5, "LP 5 dygn"), (7, "LP 7 dygn"), (14, "LP 14 dygn")):
        _, sim = day_ahead(w, ahead_days=ahead); rec(nm, sim)
    for (n, a) in RULE_N:
        _, sim, tr = run_neutral(w, du_cheap_expensive(w, n, a)); rec(("rule_n", n, a), sim)
    for k in RULE_K:
        _, sim, tr = run_neutral(w, du_price_weighted(w, k, 99)); rec(("rule_k", k), sim)
    for a in RULE_PAIR:
        _, sim, tr = run_neutral(w, du_buy_sell_pairs(w, a, 0.1, tau_h)); rec(("rule_pair", a), sim)
    for k in RULE_MD:
        _, sim, tr = run_neutral(w, du_multi_day(w, k, 99)); rec(("rule_md", k), sim)
    return (sname, wi), out


def exp_methods():
    jobs = [(s, wi) for s in SCEN for wi in range(4)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job_methods, jobs))
    pickle.dump(res, open(OUT / "methods.pkl", "wb"))


# ---------- B. band x lever heatmap (perfect foresight, comfort neutral) -----------------------
BANDS = (0.5, 1.0, 1.5, 2.5, 3.5)
LEVERS = (1, 2, 3, 4, 6, 8, 12)
CAPS = (40.0,)


def job_heat(args):
    wi, B, L, cap, neutral = args
    cfg = Cfg(band_lo=19.5 - B, band_hi=19.5 + B, u_lo=18 - L, u_hi=18 + L, flow_cap=cap, neutral=neutral)
    w = world(wi, cfg); base = w.baseline()
    u, sim = perfect_foresight(w)
    m = summary(w, sim, base, float(w.price.mean()))
    return (wi, B, L, cap, neutral), {k: m[k] for k in ("saving", "saving_neutral", "saving_pct", "saving_neutral_pct", "base_cost", "kwh", "base_kwh", "mean_room")}


def exp_heat():
    jobs = [(wi, B, L, cap, nt) for wi in range(4) for B in BANDS for L in LEVERS for cap in CAPS for nt in (True, False)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job_heat, jobs, chunksize=4))
    pickle.dump(res, open(OUT / "heat.pkl", "wb"))


# ---------- B2. how strict is 'same comfort'? --------------------------------------------------
def job_window(args):
    wi, sn, days = args
    cfg = Cfg(**{**SCEN[sn].__dict__, "neutral_days": days})
    w = world(wi, cfg); base = w.baseline(); pbar = float(w.price.mean())
    _, s1 = perfect_foresight(w)
    _, s2 = day_ahead(w)
    out = {}
    for nm, sim in (("perfekt", s1), ("dygn", s2)):
        m = summary(w, sim, base, pbar)
        out[nm] = {k: m[k] for k in ("saving", "saving_neutral", "saving_neutral_pct", "base_cost", "mean_room", "min_room", "max_room")}
    return (wi, sn, days), out


def exp_window():
    jobs = [(wi, sn, d) for wi in range(4) for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann") for d in (1, 2, 3, 7, 14, 0)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job_window, jobs))
    pickle.dump(res, open(OUT / "window.pkl", "wb"))


# ---------- C. inertia sweep ------------------------------------------------------------------
TAUS = (0.25, 0.5, 1, 2, 4, 8, 16)
TSCEN = {"Idag": Cfg(), "Utvidgat": SCEN["S3 Nedsänkning"], "Teoretiskt": SCEN["S4 Stort spann"]}


def job_tau(args):
    wi, sn, ts, mode = args
    cfg = Cfg(**{**TSCEN[sn].__dict__, "tau_scale": ts})
    w = world(wi, cfg); base = w.baseline(); pbar = float(w.price.mean())
    if mode == "perfekt":
        _, sim = perfect_foresight(w)
    elif mode == "dygn":
        _, sim = day_ahead(w)
    else:
        _, sim, _ = run_neutral(w, du_cheap_expensive(w, 6, 2.0))
    m = summary(w, sim, base, pbar)
    return (wi, sn, ts, mode), {k: m[k] for k in ("saving", "saving_neutral", "saving_neutral_pct", "base_cost")} | {"tau_h": 1.0 / (w.ks + w.ko)}


def exp_tau():
    jobs = [(wi, sn, ts, md) for wi in range(4) for sn in TSCEN for ts in TAUS for md in ("perfekt", "dygn", "regel")]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job_tau, jobs, chunksize=2))
    pickle.dump(res, open(OUT / "tau.pkl", "wb"))


# ---------- D. comfort-weight Pareto (free LP with a price on temperature) --------------------
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 1.5, 3.0)  # x (kWh per C h x mean price)


def job_pareto(args):
    wi, sn, lam = args
    cfg = Cfg(**{**SCEN[sn].__dict__, "neutral": False})
    w = world(wi, cfg); base = w.baseline(); pbar = float(w.price.mean())
    u, _ = optimise(w, cold_weight=lam * w.kwh_per_c_hour() * pbar)
    sim = w.simulate(u)
    m = summary(w, sim, base, pbar)
    return (wi, sn, lam), {k: m[k] for k in ("saving", "saving_pct", "saving_neutral", "mean_room", "min_room", "max_room", "base_cost")}


def exp_pareto():
    jobs = [(wi, sn, lam) for wi in range(4) for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann") for lam in LAMBDAS]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job_pareto, jobs))
    pickle.dump(res, open(OUT / "pareto.pkl", "wb"))


# ---------- E. robustness: model gain, placebo prices, terminal weight -------------------------
def job_robust(args):
    wi, sn, kind, val = args
    cfg = SCEN[sn]
    price = None
    if kind == "gain":
        cfg = Cfg(**{**cfg.__dict__, "gain_scale": val})
    w0 = world(wi, cfg)
    if kind == "placebo":  # optimise on another winter's prices, pay this winter's
        other = world((wi + int(val)) % 4, cfg)
        n = min(w0.n, other.n)
        fake = w0.price.copy(); fake[:n] = other.price[:n]
        wf = World(w0.hours, fake, w0.t, w0.sun, cfg)
        u, _ = optimise(wf)
        sim = w0.simulate(u)
    elif kind == "term":
        _, sim = day_ahead(w0, terminal_mult=val)
    else:
        _, sim = perfect_foresight(w0)
    base = w0.baseline(); m = summary(w0, sim, base, float(w0.price.mean()))
    return (wi, sn, kind, val), {k: m[k] for k in ("saving", "saving_neutral", "saving_neutral_pct", "base_cost", "mean_room")}


def exp_robust():
    jobs = []
    for wi in range(4):
        for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann"):
            jobs += [(wi, sn, "gain", g) for g in (0.6, 0.8, 1.0, 1.25)]
            jobs += [(wi, sn, "placebo", sh) for sh in (1, 2, 3)]
            jobs += [(wi, sn, "term", t) for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(job_robust, jobs, chunksize=2))
    pickle.dump(res, open(OUT / "robust.pkl", "wb"))


# ---------- F. example traces for the plots ---------------------------------------------------
def exp_examples():
    out = {}
    for sn in ("S1 Idag", "S3 Nedsänkning", "S4 Stort spann"):
        w = world(3, SCEN[sn]); base = w.baseline()
        u_pf, sim_pf = perfect_foresight(w)
        u_da, sim_da = day_ahead(w)
        u_r, sim_r, _ = run_neutral(w, du_cheap_expensive(w, 6, 2.0))
        out[sn] = dict(hours=w.hours, price=w.price, t_out=w.t, base_u=w.base_eff, base=base, pf=(u_pf, sim_pf),
                       da=(u_da, sim_da), rule=(u_r, sim_r), flow_ref=w.f_ref, cfg=SCEN[sn],
                       flow_pf=np.minimum(w.flow_pump(u_pf), SCEN[sn].flow_cap))
    pickle.dump(out, open(OUT / "examples.pkl", "wb"))


# ---------- G. model validation ----------------------------------------------------------------
def exp_validation():
    cfg = Cfg()
    a, b = WIN[0][1], WIN[-1][2]
    h, s = window(HOURS, SER, a, b)
    w = World(h, s["price"], s["t_out"], s["sun"], cfg)
    rts = s["rts"]
    ok = np.isfinite(rts) & np.isfinite(s["comp"]) & np.isfinite(s["aux"]) & (s["hot"] < 0.5)
    comp, aux = w.energy(np.where(np.isfinite(rts), rts, 18.0))
    pred, meas = comp + aux, s["comp"] + s["aux"]
    day = local_days(w)
    rows = []
    for d in np.unique(day):
        m = (day == d) & ok
        if m.sum() >= 14:
            rows.append((d, pred[m].sum() * 24 / m.sum(), meas[m].sum() * 24 / m.sum(), float(np.mean(w.t[day == d]))))
    rows = np.array(rows)
    pickle.dump({"days": rows}, open(OUT / "validation.pkl", "wb"))


if __name__ == "__main__":
    todo = sys.argv[1:] or ["validation", "methods", "heat", "tau", "pareto", "robust", "examples"]
    for name in todo:
        t0 = time.time()
        globals()["exp_" + name]()
        print(name, "done in %.0fs" % (time.time() - t0), flush=True)
