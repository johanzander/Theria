"""Control methods compared in the study: rule-based, bess-style pairs, day-ahead LP, perfect-foresight LP."""

from __future__ import annotations

import numpy as np

from sp_engine import TZ, World, optimise


def local_days(w: World) -> np.ndarray:
    return np.array([h.astimezone(TZ).date().toordinal() for h in w.hours])


def settle(w: World, sim: dict, mean_price: float) -> float:
    """SEK owed to bring the room back to the baseline comfort: colder mean / colder end, at the
    price of the energy it takes to hold or to store that heat. Warmer than baseline earns nothing."""
    target = w.cfg.target
    block = w.n if w.cfg.neutral_days == 0 else 24 * w.cfg.neutral_days
    deficit = target - sim["room"]
    # a warm week does not pay for a cold one: every block is settled on its own
    cold_hours = sum(max(0.0, float(deficit[a:a + block].sum())) for a in range(0, w.n, block))
    cold_end = max(0.0, target - float(sim["room"][-1]))
    return mean_price * (w.kwh_per_c_hour() * cold_hours + w.kwh_per_c_stock() * cold_end)


def summary(w: World, sim: dict, base: dict, mean_price: float) -> dict:
    """Costs and comfort of one run relative to the baseline."""
    # any comfort shortfall is billed at the season's 90th-percentile price: no method gains by it
    cost, settled = float(sim["cost"].sum()), float(sim["cost"].sum()) + settle(w, sim, float(np.percentile(w.price, 90)))
    room = sim["room"]
    lo, hi = w.cfg.band_lo, w.cfg.band_hi
    base_cost = float(base["cost"].sum())
    return {
        "cost": cost, "settled": settled, "base_cost": base_cost,
        "saving": base_cost - cost, "saving_neutral": base_cost - settled,
        "saving_pct": 100 * (base_cost - cost) / base_cost,
        "saving_neutral_pct": 100 * (base_cost - settled) / base_cost,
        "kwh": float(sim["kwh"].sum()), "base_kwh": float(base["kwh"].sum()),
        "mean_room": float(room.mean()), "min_room": float(room.min()), "max_room": float(room.max()),
        "violation_ch": float(np.sum(np.maximum(lo - room, 0) + np.maximum(room - hi, 0))),
        "avg_price_paid": float((sim["cost"].sum()) / sim["kwh"].sum()),
        "base_avg_price": float(base["cost"].sum() / base["kwh"].sum()),
    }


# --- rule-based desired deviations ------------------------------------------------------------
def du_cheap_expensive(w: World, n_hours: int, adj: float) -> np.ndarray:
    """Theria's current idea: heat more in the N cheapest hours of the local day, less in the N dearest."""
    du, day = np.zeros(w.n), local_days(w)
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        order = idx[np.argsort(w.price[idx], kind="stable")]
        du[order[:n_hours]] += adj
        du[order[-n_hours:]] -= adj
    return du


def du_price_weighted(w: World, weight: float, cap: float) -> np.ndarray:
    """Setpoint offset proportional to the hour's price z-score within the local day."""
    du, day = np.zeros(w.n), local_days(w)
    for d in np.unique(day):
        idx = np.where(day == d)[0]
        p = w.price[idx]
        z = (p - p.mean()) / max(float(p.std()), 0.05)
        du[idx] = np.clip(-weight * z, -cap, cap)
    return du


def du_buy_sell_pairs(w: World, adj: float, margin: float, tau_h: float) -> np.ndarray:
    """bess-manager style: pair a cheap 'charge' hour with a later dear 'discharge' hour while the
    spread beats the margin after the heat has decayed on its way (exp(-dt/tau))."""
    du, day = np.zeros(w.n), local_days(w)
    for d in np.unique(day):
        idx = list(np.where(day == d)[0])
        free = set(idx)
        while True:
            best = None
            for b in sorted(free, key=lambda i: w.price[i]):
                later = [s for s in free if s > b]
                if not later:
                    continue
                s = max(later, key=lambda i: w.price[i])
                gain = w.price[s] * np.exp(-(s - b) / tau_h) - w.price[b]
                if gain > margin:
                    best = (b, s)
                    break
            if best is None:
                break
            b, s = best
            du[b] += adj
            du[s] -= adj
            free -= {b, s}
    return du


def run_policy(w: World, du: np.ndarray, trim: float = 0.0):
    """Apply desired setpoint deviations through a band guard; returns (u, sim)."""
    c = w.cfg
    n = w.n
    u = np.empty(n)
    R = np.empty(n)
    r, s = c.target, w.s_ref[0]
    for i in range(n):
        want = float(np.clip(w.base_eff[i] + du[i] + trim, w.u_min[i], w.u_max[i]))
        r1, _ = w.step(i, r, s, want)
        # two-hour look: the slab hands heat to the room one hour later; keep the room inside its band
        if r1 > c.band_hi - 0.05 and want > w.base_eff[i]:
            want = float(w.base_eff[i])
        if r1 < c.band_lo + 0.05 and want < w.base_eff[i]:
            want = float(w.base_eff[i])
        u[i] = want
        r, s = w.step(i, r, s, want)
        R[i] = r
    comp, aux = w.energy(u)
    return u, {"room": R, "comp": comp, "aux": aux, "kwh": comp + aux, "cost": w.price * (comp + aux)}


def run_neutral(w: World, du: np.ndarray):
    """Run a rule with a constant trim chosen so the mean room temperature equals the target."""
    lo, hi = -4.0, 4.0
    best = None
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        u, sim = run_policy(w, du, mid)
        best = (u, sim, mid)
        if sim["room"].mean() < w.cfg.target:
            lo = mid
        else:
            hi = mid
    return best


# --- LP methods -------------------------------------------------------------------------------
def perfect_foresight(w: World):
    """The optimum with every price known and every comfort block held at the baseline mean."""
    u, res = optimise(w)
    return u, w.simulate(u)


def day_ahead(w: World, terminal_mult: float = 1.0, plan_hour: int = 14, ahead_days: int = 1):
    """Rolling LP: every day at `plan_hour` local, plan with the prices known by then (to the end of
    `ahead_days` days after today), execute 24 h, re-plan. Every comfort block must still average the
    target (hours already lived count), and heat left at the horizon end is valued at the trailing
    7-day price."""
    day = local_days(w)
    hour = np.array([h.astimezone(TZ).hour for h in w.hours])
    starts = [0] + [i for i in range(1, w.n) if hour[i] == plan_hour]
    block = w.n if w.cfg.neutral_days == 0 else 24 * w.cfg.neutral_days
    exec_sums: dict[int, float] = {}
    u_all = np.empty(w.n)
    r, s = w.cfg.target, w.s_ref[0]
    for k, j in enumerate(starts):
        j_end = starts[k + 1] if k + 1 < len(starts) else w.n
        h_end = int(np.searchsorted(day, day[j] + ahead_days, side="right"))
        h_end = max(h_end, j_end + 1) if j_end < w.n else w.n
        h_end = min(h_end, w.n)
        sub = w.slice(j, h_end)
        pbar = float(np.mean(w.price[max(0, j - 168):max(j, 1)])) if j > 0 else float(np.mean(w.price[:24]))
        v = terminal_mult * w.kwh_per_c_stock() * max(pbar, 0.0)
        x0, y0 = r - w.cfg.target, s - w.s_ref[j]
        for margin in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):  # a fast house can be pushed out of its band by the previous plan's end state
            try:
                u_sub, _ = optimise(sub, terminal_value=v, x0=x0, y0=y0, end_floor=False, neutral=True,
                                    band_margin=margin, exec_sums=exec_sums,
                                    block_penalty=2.0 * w.kwh_per_c_hour() * float(np.max(np.maximum(sub.price, 0.0))))
                break
            except RuntimeError:
                if margin == 4.0:
                    raise
        for i in range(j, j_end):
            u_all[i] = u_sub[i - j]
            r, s = w.step(i, r, s, u_all[i])
            exec_sums[i // block] = exec_sums.get(i // block, 0.0) + (r - w.cfg.target)
    return u_all, w.simulate(u_all)


def du_multi_day(w: World, weight: float, cap: float, half_window_h: int = 48) -> np.ndarray:
    """Price z-score against the centred multi-day mean (needs a price forecast): matched to a slow house."""
    p = w.price
    n = len(p)
    du = np.zeros(n)
    cs = np.concatenate([[0.0], np.cumsum(p)])
    cs2 = np.concatenate([[0.0], np.cumsum(p * p)])
    for i in range(n):
        a, b = max(0, i - half_window_h), min(n, i + half_window_h + 1)
        m = (cs[b] - cs[a]) / (b - a)
        v = (cs2[b] - cs2[a]) / (b - a) - m * m
        du[i] = np.clip(-weight * (p[i] - m) / max(np.sqrt(max(v, 0.0)), 0.05), -cap, cap)
    return du
