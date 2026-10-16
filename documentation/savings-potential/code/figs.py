"""Figures and numbers for the savings-potential report. Reads results/*.pkl."""

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from sp_engine import SCRATCH, TZ, Cfg, World, load_study, winters, window

RES = SCRATCH / "results"
OUT = SCRATCH / "figs"
OUT.mkdir(exist_ok=True)

# light-mode tokens of the validated reference palette (dataviz skill, references/palette.md)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898880", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlelocation": "left", "font.size": 10.5,
    "legend.frameon": False, "lines.linewidth": 2.0, "axes.axisbelow": True, "figure.dpi": 110,
})

SCEN_ORDER = ["S1 Idag", "S2 Brett band", "S3 Nedsänkning", "S4 Stort spann"]
SCEN_LABEL = {
    "S1 Idag": "S1  Idag\nband 18–21 °C\nbörvärde 16–20, shunt 40 °C",
    "S2 Brett band": "S2  Brett band\nband 16–23 °C\nreglage som idag",
    "S3 Nedsänkning": "S3  Nedsänkning\nband 16–23 °C\nbörvärde 8–20, flöde ≤ 40 °C",
    "S4 Stort spann": "S4  Stort spann\nband 16–23 °C\nbörvärde 8–26, flöde ≤ 40 °C",
}
SCEN_SHORT = {k: k.split(" ", 1)[0] for k in SCEN_ORDER}
WINTERS = [w[0] for w in winters()]


def load(name):
    return pickle.load(open(RES / f"{name}.pkl", "rb"))


def save(fig, name):
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pooled(R, sn, key, field="saving_neutral"):
    tot = sum(R[(sn, wi)][key][field] for wi in range(4))
    base = sum(R[(sn, wi)][key]["base_cost"] for wi in range(4))
    return tot, 100 * tot / base, base


def best_of(R, sn, family):
    keys = [k for k in R[(sn, 0)] if isinstance(k, tuple) and k[0] == family]
    return max(keys, key=lambda k: pooled(R, sn, k)[0])


def bootstrap(R, sn, key, n_boot=3000, block=7, seed=7):
    """Block bootstrap (weeks) of the comfort-neutral saving, pooled over the four winters."""
    rng = np.random.default_rng(seed)
    days = np.concatenate([R[(sn, wi)][key]["daily_saving"] - R[(sn, wi)][key]["settle"] / len(R[(sn, wi)][key]["daily_saving"]) for wi in range(4)])
    base = np.concatenate([R[(sn, wi)]["base_daily"] for wi in range(4)])
    n = len(days)
    nb = int(np.ceil(n / block))
    starts_max = n - block
    pct = np.empty(n_boot)
    sek = np.empty(n_boot)
    for b in range(n_boot):
        st = rng.integers(0, starts_max + 1, nb)
        idx = (st[:, None] + np.arange(block)[None, :]).ravel()[:n]
        pct[b] = 100 * days[idx].sum() / base[idx].sum()
        sek[b] = days[idx].sum() / 4
    return (np.percentile(pct, [2.5, 97.5]), np.percentile(sek, [2.5, 97.5]))


# ---------------------------------------------------------------------------------------------
def fig_inertia():
    """The house as a low-pass filter: step response and attenuation of a daily setpoint swing."""
    import sys
    sys.path.insert(0, "/Users/johanzander/GitHub/Theria")
    hours, ser = load_study()
    lab, a, b = winters()[3]
    h, s = window(hours, ser, a, b)
    houses = {"Golvvärme (Johans bottenvåning), τ ≈ 5 dygn": 1.0, "Lättare hus, τ ≈ 1 dygn": 4.95, "Radiatorer (Butiken), τ ≈ 12,5 h": 9.5}
    colors = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    t = np.arange(0, 24 * 12 + 1)
    for (name, ts), col in zip(houses.items(), colors):
        w = World(h[:10], s["price"][:10], s["t_out"][:10], s["sun"][:10], Cfg(tau_scale=ts, flow_cap=1e9))
        gain = w.ks / (w.ks + w.ko)
        rate = w.ks + w.ko
        # +1 degree of setpoint = +1.9 C of flow: the room settles at gain*1.9 above where it was
        resp = gain * 1.9 * (1 - np.exp(-rate * t))
        ax.plot(t / 24, resp, color=col, label=name)
    ax.set_xlabel("dygn efter höjt börvärde (+1 °C)")
    ax.set_ylabel("rumstemperatur, höjning (°C)")
    ax.set_title("Stegsvar: hur fort huset följer")
    ax.legend(loc="lower right")
    ax = axes[1]
    periods = np.array([3, 6, 12, 24, 48, 96, 168, 336])
    for (name, ts), col in zip(houses.items(), colors):
        w = World(h[:10], s["price"][:10], s["t_out"][:10], s["sun"][:10], Cfg(tau_scale=ts, flow_cap=1e9))
        gain = w.ks / (w.ks + w.ko)
        rate = w.ks + w.ko
        om = 2 * np.pi / periods
        amp = gain * 1.9 * rate / np.sqrt(rate ** 2 + om ** 2)
        ax.plot(periods, amp, color=col, marker="o", ms=4)
    ax.set_xscale("log")
    ax.set_xticks(periods)
    ax.set_xticklabels(["3 h", "6 h", "12 h", "1 d", "2 d", "4 d", "7 d", "14 d"])
    ax.axvline(24, color=MUTED, lw=1, ls="--")
    ax.text(25, ax.get_ylim()[1] * 0.93, "prisets dygnscykel", color=INK2, fontsize=9)
    ax.set_xlabel("periodtid för börvärdets svängning")
    ax.set_ylabel("rumssvängning per °C börvärdessvängning (°C)")
    ax.set_title("Dämpning: korta pristoppar försvinner i golvet")
    fig.tight_layout()
    save(fig, "01_tröghet.png")


def fig_validation():
    V = load("validation")["days"]
    pred, meas, tout = V[:, 1], V[:, 2], V[:, 3]
    r2 = np.corrcoef(pred, meas)[0, 1] ** 2
    fig, ax = plt.subplots(figsize=(5.6, 5))
    sc = ax.scatter(meas, pred, c=tout, cmap="Blues_r", s=14, alpha=0.9, edgecolors="none")
    m = max(meas.max(), pred.max()) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=1, ls="--")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("uppmätt elförbrukning värme (kWh/dygn)")
    ax.set_ylabel("modellens förbrukning (kWh/dygn)")
    ax.set_title(f"Pumpmodellen mot mätdata\n{len(pred)} dygn, R² = {r2:.2f}, bias {100*np.mean(pred-meas)/np.mean(meas):+.1f} %", fontsize=11)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("medel utetemp (°C)")
    save(fig, "02_validering.png")
    return {"days": int(len(pred)), "r2": float(r2), "bias_pct": float(100 * np.mean(pred - meas) / np.mean(meas)),
            "mae_kwh": float(np.mean(np.abs(pred - meas)))}


METHODS = [
    ("LP perfekt", "Tak: optimum med känd framtid", BLUE),
    ("LP dygnet före", "Praktiskt: optimerar dygnet före", ORANGE),
    ("rule_md", "Regel, flerdygns pris (kräver prisprognos)", AQUA),
    ("rule_k", "Regel, prisviktad inom dygnet", YELLOW),
    ("rule_n", "Regel: N billigaste/dyraste timmar (Theria idag)", MAGENTA),
    ("rule_pair", "Köp–sälj-par (bess-manager-stil)", GREEN),
]


def fig_methods(R):
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    width = 0.13
    summary = {}
    for si, sn in enumerate(SCEN_ORDER):
        for mi, (key, label, col) in enumerate(METHODS):
            k = key if isinstance(key, str) and key.startswith("LP") else best_of(R, sn, key)
            tot, pct, base = pooled(R, sn, k)
            ci_pct, ci_sek = bootstrap(R, sn, k)
            x = si + (mi - (len(METHODS) - 1) / 2) * (width + 0.01)
            ax.bar(x, pct, width, color=col, label=label if si == 0 else None)
            ax.errorbar(x, pct, yerr=[[pct - ci_pct[0]], [ci_pct[1] - pct]], color=INK2, lw=1, capsize=2)
            ax.text(x, ci_pct[1] + 0.6, f"{pct:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
            summary.setdefault(sn, {})[label] = dict(pct=pct, sek_per_winter=tot / 4, ci_pct=list(ci_pct), ci_sek=list(ci_sek),
                                                    param=str(k[1:]) if isinstance(k, tuple) else "")
    ax.set_xticks(range(len(SCEN_ORDER)))
    ax.set_xticklabels([SCEN_LABEL[s] for s in SCEN_ORDER], fontsize=9)
    ax.set_ylabel("besparing av spotkostnaden för värme (%)")
    ax.set_title("Besparing per metod och spelrum  (4 vintrar 2022/23–2025/26, varje vecka samma medeltemperatur som idag)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_ylim(0, 42)
    save(fig, "03_metoder.png")
    return summary


def fig_heat(H):
    caps = [(40.0, "Flöde ≤ 40 °C (shuntarna, hårt krav)")]
    bands = (0.5, 1.0, 1.5, 2.5, 3.5)
    levers = (1, 2, 3, 4, 6, 8, 12)
    fig, axes = plt.subplots(1, 1, figsize=(7.5, 4.6), sharey=True)
    axes = [axes]
    vmax = 0
    grids = []
    for cap, _ in caps:
        g = np.zeros((len(bands), len(levers)))
        for bi, B in enumerate(bands):
            for li, L in enumerate(levers):
                tot = sum(H[(wi, B, L, cap, True)]["saving_neutral"] for wi in range(4))
                base = sum(H[(wi, B, L, cap, True)]["base_cost"] for wi in range(4))
                g[bi, li] = 100 * tot / base
        grids.append(g)
        vmax = max(vmax, g.max())
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, g, (cap, title) in zip(axes, grids, caps):
        im = ax.imshow(g, origin="lower", cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(levers))); ax.set_xticklabels([f"±{l}" for l in levers])
        ax.set_yticks(range(len(bands))); ax.set_yticklabels([f"±{b:g}  ({19.5-b:g}–{19.5+b:g})" for b in bands])
        for bi in range(len(bands)):
            for li in range(len(levers)):
                v = g[bi, li]
                ax.text(li, bi, f"{v:.0f}", ha="center", va="center", fontsize=9, color="white" if v > vmax * 0.55 else INK)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("reglagets spelrum kring 18 °C börvärde")
        ax.grid(False)
    axes[0].set_ylabel("tillåten rumstemperatur (°C)")
    fig.suptitle("Teoretisk besparing (%) vid perfekt framsyn — komfortband × reglagets spelrum", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "04_band_x_reglage.png")
    return {f"{cap}": g.tolist() for (cap, _), g in zip(caps, grids)}


def pick_window(d, days=14):
    """Start day of the 14-day window with the largest saving among those that start and end at the same room temperature."""
    best, best_v = 0, -1e9
    base_c = d["base"]["cost"]
    room = d["pf"][1]["room"]
    for t0 in range(5, len(base_c) // 24 - days - 2):
        a, b = t0 * 24, (t0 + days) * 24
        if abs(room[b - 1] - room[a - 1]) > 0.25:
            continue
        v = base_c[a:b].sum() - d["pf"][1]["cost"][a:b].sum()
        if v > best_v:
            best, best_v = t0, v
    return best


def fig_example(E, sn, fname, days=14):
    d = E[sn]
    t0_day = pick_window(d, days)
    hrs = np.array([h.astimezone(TZ) for h in d["hours"]])
    i0 = t0_day * 24
    i1 = i0 + days * 24
    sl = slice(i0, i1)
    x = hrs[sl]
    cfg = d["cfg"]
    fig, axes = plt.subplots(5, 1, figsize=(12.5, 11.5), sharex=True, gridspec_kw={"height_ratios": [1.1, 0.8, 1.2, 1.5, 1.0]})
    ax = axes[0]
    ax.step(x, d["price"][sl], where="post", color=INK2, lw=1.3)
    ax.set_ylabel("spotpris (SEK/kWh)")
    ax.set_title("Pris, väder och optimal styrning – 14 dygn ur vintern 2025/26 (praktisk plan slutar kallare; underskottet är inte avräknat i kurvan)", loc="left")
    ax = axes[1]
    ax.plot(x, d["t_out"][sl], color=BLUE, lw=1.6)
    ax.set_ylabel("utetemp (°C)")
    ax = axes[2]
    ax.step(x, d["base_u"][sl], where="post", color=MUTED, lw=1.4, label="baslinje (fast börvärde)")
    ax.step(x, d["pf"][0][sl], where="post", color=BLUE, lw=1.6, label="optimalt, känd framtid")
    ax.step(x, d["da"][0][sl], where="post", color=ORANGE, lw=1.4, label="praktiskt, dygnet före")
    ax.set_ylabel("rumsbörvärde (°C)")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    ax = axes[3]
    ax.axhspan(cfg.band_lo, cfg.band_hi, color=SEQ[0], alpha=0.5, lw=0)
    ax.text(x[0], cfg.band_hi - 0.05, " tillåtet komfortband", va="top", fontsize=9, color=INK2)
    ax.plot(x, d["base"]["room"][sl] if False else np.full(len(x), cfg.target), color=MUTED, lw=1.4, label="baslinje")
    ax.plot(x, d["pf"][1]["room"][sl], color=BLUE, lw=1.8, label="optimalt, känd framtid")
    ax.plot(x, d["da"][1]["room"][sl], color=ORANGE, lw=1.5, label="praktiskt, dygnet före")
    ax.set_ylabel("rumstemperatur (°C)")
    ax.set_ylim(cfg.band_lo - 0.4, cfg.band_hi + 0.4)
    ax.legend(loc="lower left", ncol=3, fontsize=9)
    ax = axes[4]
    base_c = np.cumsum(d["base"]["cost"][sl])
    ax.plot(x, base_c - np.cumsum(d["pf"][1]["cost"][sl]), color=BLUE, label="optimalt, känd framtid")
    ax.plot(x, base_c - np.cumsum(d["da"][1]["cost"][sl]), color=ORANGE, label="praktiskt, dygnet före")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_ylabel("ackumulerad besparing (SEK)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b", tz=TZ))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2, tz=TZ))
    fig.tight_layout()
    save(fig, fname)


def fig_tau(T):
    taus = (0.25, 0.5, 1, 2, 4, 8, 16)
    sn_list = [("Idag", "Idag: börvärde 16–20, band 18–21"), ("Utvidgat", "S3 Nedsänkning: börvärde 8–20, band 16–23, flöde ≤ 40"), ("Teoretiskt", "S4 Stort spann: börvärde 8–26, band 16–23, flöde ≤ 40")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=False)
    modes = [("perfekt", "känd framtid (optimum)", BLUE), ("dygn", "optimerar dygnet före", ORANGE), ("regel", "regel 6 h × 2 °C", YELLOW)]
    for ax, (sn, title) in zip(axes, sn_list):
        for md, label, col in modes:
            ys, xs = [], []
            for ts in taus:
                tot = sum(T[(wi, sn, ts, md)]["saving_neutral"] for wi in range(4))
                base = sum(T[(wi, sn, ts, md)]["base_cost"] for wi in range(4))
                ys.append(100 * tot / base)
                xs.append(T[(0, sn, ts, md)]["tau_h"] / 24)
            ax.plot(xs, ys, color=col, marker="o", ms=4, label=label)
        ax.set_xscale("log")
        ax.axvline(0.52, color=MUTED, lw=1, ls="--"); ax.axvline(4.95, color=MUTED, lw=1, ls="--")
        ax.text(0.54, 0.97, "Butiken\n(radiatorer)", transform=ax.get_xaxis_transform(), va="top", fontsize=8.5, color=INK2)
        ax.text(5.2, 0.97, "Johans\ngolvvärme", transform=ax.get_xaxis_transform(), va="top", fontsize=8.5, color=INK2)
        ax.set_xlabel("husets tidskonstant (dygn)")
        ax.set_title(title, fontsize=10.5)
        ax.set_xticks([0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0,3", "1", "3", "10", "30"])
    axes[0].set_ylabel("besparing (%), samma komfort")
    axes[0].legend(loc="center left", fontsize=9)
    fig.suptitle("Värdet av husets tröghet: besparing mot tidskonstant", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "06_tröghet_värde.png")


def fig_horizon(R):
    names = [("LP dygnet före", 1), ("LP 2 dygn", 2), ("LP 3 dygn", 3), ("LP 5 dygn", 5), ("LP 7 dygn", 7), ("LP 14 dygn", 14)]
    cols = {"S1 Idag": BLUE, "S3 Nedsänkning": ORANGE, "S4 Stort spann": AQUA}
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for sn, col in cols.items():
        xs = [d for _, d in names]
        ys = [pooled(R, sn, k)[1] for k, _ in names]
        ax.plot(xs, ys, color=col, marker="o", ms=4)
        perf = pooled(R, sn, "LP perfekt")[1]
        ax.axhline(perf, color=col, lw=1, ls=":")
        ax.text(14.3, ys[-1], SCEN_SHORT[sn], color=INK2, va="center", fontsize=9)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 7, 14]); ax.set_xticklabels(["1", "2", "3", "5", "7", "14"])
    ax.set_xlabel("antal dygn framåt som priset är känt (prisprognos)")
    ax.set_ylabel("besparing (%)")
    ax.set_title("Vad är en längre prisprognos värd?  (prickat = hela vintern känd)")
    ax.set_xlim(0.9, 18)
    save(fig, "07_prognoshorisont.png")


def fig_window(W):
    cols = {"S1 Idag": BLUE, "S3 Nedsänkning": ORANGE, "S4 Stort spann": AQUA}
    days = (1, 2, 3, 7, 14, 0)
    xs = [1, 2, 3, 7, 14, 40]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for sn, col in cols.items():
        for md, ls in (("perfekt", "-"), ("dygn", "--")):
            ys = []
            for d in days:
                tot = sum(W[(wi, sn, d)][md]["saving_neutral"] for wi in range(4))
                base = sum(W[(wi, sn, d)][md]["base_cost"] for wi in range(4))
                ys.append(100 * tot / base)
            ax.plot(xs, ys, color=col, ls=ls, marker="o", ms=4, label=f"{SCEN_SHORT[sn]} " + ("känd framtid" if md == "perfekt" else "dygnet före"))
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(["1 dygn", "2", "3", "7 (vår standard)", "14", "hela vintern"], fontsize=9)
    ax.set_xlabel("hur strikt 'samma komfort' är: medeltemperaturen ska nå målet varje …")
    ax.set_ylabel("besparing (%)")
    ax.set_title("Komfortkravets längd avgör hur mycket som kan flyttas")
    ax.legend(fontsize=8.5, ncol=2, loc="upper left")
    save(fig, "11_komfortfönster.png")


def fig_pareto(P):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    cols = {"S1 Idag": BLUE, "S3 Nedsänkning": ORANGE, "S4 Stort spann": AQUA}
    lams = sorted({k[2] for k in P})
    for sn, col in cols.items():
        xs, ys = [], []
        for lam in lams:
            tot = sum(P[(wi, sn, lam)]["saving"] for wi in range(4))
            base = sum(P[(wi, sn, lam)]["base_cost"] for wi in range(4))
            xs.append(np.mean([P[(wi, sn, lam)]["mean_room"] for wi in range(4)]))
            ys.append(100 * tot / base)
        ax.plot(xs, ys, color=col, marker="o", ms=4, label=SCEN_SHORT[sn] + " " + sn.split(" ", 1)[1])
    ax.axvline(19.5, color=MUTED, lw=1, ls="--")
    ax.text(19.52, ax.get_ylim()[1] * 0.97, " baslinjens\n medeltemp", fontsize=8.5, color=INK2, va="top")
    ax.set_xlabel("medelrumstemperatur över vintern (°C)")
    ax.set_ylabel("besparing av spotkostnaden (%)")
    ax.set_title("Vikten på komfort: ju kallare medeltemp, desto större 'besparing'")
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "08_komfortvikt.png")


def fig_rules(R):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    Ns = (2, 4, 6, 8, 12)
    adjs = (0.5, 1, 2, 3, 4, 6)
    for ax, sn in zip(axes, ["S1 Idag", "S3 Nedsänkning", "S4 Stort spann"]):
        g = np.array([[pooled(R, sn, ("rule_n", n, a))[1] for a in adjs] for n in Ns])
        im = ax.imshow(g, origin="lower", cmap=cmap, vmin=0, vmax=max(g.max(), 1), aspect="auto")
        ax.set_xticks(range(len(adjs))); ax.set_xticklabels([f"{a:g}" for a in adjs])
        ax.set_yticks(range(len(Ns))); ax.set_yticklabels([str(n) for n in Ns])
        for i in range(len(Ns)):
            for j in range(len(adjs)):
                ax.text(j, i, f"{g[i,j]:.1f}", ha="center", va="center", fontsize=8.5, color="white" if g[i, j] > g.max() * 0.55 else INK)
        ax.set_title(SCEN_SHORT[sn] + " " + sn.split(" ", 1)[1], fontsize=11)
        ax.set_xlabel("justering (°C börvärde)")
        ax.grid(False)
    axes[0].set_ylabel("antal billigaste / dyraste timmar per dygn")
    fig.suptitle("Regelvikter: besparing (%) för 'N billigaste/dyraste timmar' (komfortneutral)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "09_regelvikter.png")


def fig_robust(X):
    sns = ["S1 Idag", "S3 Nedsänkning", "S4 Stort spann"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    def pct(sn, kind, val, f="saving_neutral"):
        tot = sum(X[(wi, sn, kind, val)][f] for wi in range(4))
        base = sum(X[(wi, sn, kind, val)]["base_cost"] for wi in range(4))
        return 100 * tot / base
    cols = {"S1 Idag": BLUE, "S3 Nedsänkning": ORANGE, "S4 Stort spann": AQUA}
    ax = axes[0]
    gains = (0.6, 0.8, 1.0, 1.25)
    for sn in sns:
        ax.plot(gains, [pct(sn, "gain", g) for g in gains], color=cols[sn], marker="o", ms=4, label=SCEN_SHORT[sn])
    ax.set_xlabel("antagen golvgain (× skattad)"); ax.set_ylabel("besparing (%)")
    ax.set_title("Modellkänslighet: hur starkt golvet svarar", fontsize=11)
    ax.legend(fontsize=9)
    ax = axes[1]
    shifts = (1, 2, 3)
    w = 0.25
    for i, sn in enumerate(sns):
        real = pct(sn, "gain", 1.0)
        plc = [pct(sn, "placebo", s) for s in shifts]
        ax.bar(i - 0.22, real, 0.2, color=cols[sn])
        ax.bar(i + 0.0, np.mean(plc), 0.2, color=MUTED)
        ax.text(i - 0.22, real + 0.4, f"{real:.1f}", ha="center", fontsize=8.5, color=INK2)
        ax.text(i + 0.0, max(np.mean(plc), 0) + 0.4, f"{np.mean(plc):.1f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(range(3)); ax.set_xticklabels([SCEN_SHORT[s] for s in sns])
    ax.set_ylabel("besparing (%)")
    ax.set_title("Placebo: plan byggd på fel års priser (grå)", fontsize=11)
    ax = axes[2]
    ts = (0.0, 0.5, 1.0, 1.5, 2.0)
    for sn in sns:
        ax.plot(ts, [pct(sn, "term", t) for t in ts], color=cols[sn], marker="o", ms=4, label=SCEN_SHORT[sn])
    ax.set_xlabel("värde på kvarvarande värme vid planens slut (× beräknat)")
    ax.set_ylabel("besparing (%), dygnet före")
    ax.set_title("Slutvärde: vikt på lagrad värme", fontsize=11)
    fig.tight_layout()
    save(fig, "10_robusthet.png")


if __name__ == "__main__":
    R = load("methods")
    out = {}
    fig_inertia()
    out["validation"] = fig_validation()
    out["methods"] = fig_methods(R)
    out["heat"] = fig_heat(load("heat"))
    E = load("examples")
    fig_example(E, "S1 Idag", "05a_exempel_idag.png")
    fig_example(E, "S3 Nedsänkning", "05b_exempel_utvidgat.png")
    fig_example(E, "S4 Stort spann", "05c_exempel_teoretiskt.png")
    fig_tau(load("tau"))
    fig_horizon(R)
    fig_window(load("window"))
    fig_pareto(load("pareto"))
    fig_rules(R)
    fig_robust(load("robust"))
    # per-winter table
    tab = {}
    for sn in SCEN_ORDER:
        tab[sn] = {k: [R[(sn, wi)][k]["saving_neutral_pct"] for wi in range(4)] for k in ("LP perfekt", "LP dygnet före")}
        tab[sn]["base_cost"] = [R[(sn, wi)]["LP perfekt"]["base_cost"] for wi in range(4)]
        tab[sn]["saving_sek"] = {k: [R[(sn, wi)][k]["saving_neutral"] for wi in range(4)] for k in ("LP perfekt", "LP dygnet före")}
    out["per_winter"] = tab
    json.dump(out, open(OUT / "numbers.json", "w"), indent=1, default=float)
    print("ok")
