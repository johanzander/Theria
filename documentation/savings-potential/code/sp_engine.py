"""Engine for the heating-shift savings study: data, anchored house model, LP optimiser, heuristics.

The house is Theria's own fitted model (room_model + pump_model). The room's unmodelled gains are
anchored so that the flat base setpoint holds the target exactly (as the planner does); every
policy is then judged on how it deviates from that. All optimisation is a linear programme in the
deviations (convex => global optimum); every plan is replayed through the non-linear simulator.
"""

from __future__ import annotations

import copy
import datetime as dt
import pickle
import sys

sys.path.insert(0, "/Users/johanzander/GitHub/Theria")
from dataclasses import dataclass, replace
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

SCRATCH = Path(__file__).parent
TZ = ZoneInfo("Europe/Stockholm")
COMPRESSOR_MAX_KWH = 5.0  # AirX 170 capacity diagram: 4.4-5.8 kW input at max Hz, flow 35-55 C (3.6 was the AirX 9 column)
AUX_ONSET = 44.0
NEUTRAL_SLACK_CH = 0.5  # degree-hours of slack on the 'mean room >= target' row (0.0001 C over a winter): keeps the LP non-degenerate
END_SLACK_C = 0.002


@dataclass(frozen=True)
class Cfg:
    """One experiment: comfort band, actuator authority and house inertia."""

    band_lo: float = 18.0
    band_hi: float = 21.0
    target: float = 19.5
    u_lo: float = 16.0  # room-setpoint range the control may use
    u_hi: float = 20.0
    u_base: float = 18.0
    flow_cap: float = 40.0  # shunts limit the floor loops (inf = none)
    tau_scale: float = 1.0  # >1 = faster house (rates multiplied)
    gain_scale: float = 1.0
    neutral: bool = True  # True: mean room temperature >= target (pure shifting)
    max_step: float | None = None  # max setpoint rise per hour (product: 2 C)
    max_step_down: float | None = None  # max setpoint fall per hour
    neutral_days: int = 7  # comfort-neutral block length (0 = whole window): every block must average >= target


class World:
    """Models + the hourly series of one window."""

    def __init__(self, hours, price, t_out, sun, cfg: Cfg):
        m = pickle.load(open(SCRATCH / "models.pkl", "rb"))
        self.pump, room = m["pump"], m["room"]
        self.cfg = cfg
        g = cfg.gain_scale
        ks, ko = room.k_slab, room.k_out
        # scale response gain (k_slab share) and rate (both)
        total = (ks + ko)
        gain = ks / total * g
        gain = min(gain, 0.95)
        total *= cfg.tau_scale
        self.ks, self.ko = gain * total, (1 - gain) * total
        self.a = 1.0 - np.exp(-1.0 / room.tau_slab_h)
        self.hours, self.price, self.t, self.sun = hours, price, t_out, sun
        self.n = len(hours)
        self.i0 = 0  # offset of this world inside its parent (blocks are aligned to the parent window)
        p = self.pump
        # compressor cannot deliver more than 3.6 kW: the highest setpoint the model can ask for
        self.u_cap = (COMPRESSOR_MAX_KWH - p.energy_intercept - p.energy_per_outdoor * t_out) / p.energy_per_rts
        self.base_eff = np.minimum(cfg.u_base, self.u_cap)
        self.u_max = np.minimum(cfg.u_hi, np.maximum(self.u_cap, self.base_eff))
        self.u_min = np.minimum(cfg.u_lo, self.base_eff)
        self.flow_pump_ref = self.flow_pump(self.base_eff)
        self.f_ref = np.minimum(self.flow_pump_ref, cfg.flow_cap)
        # reference slab trajectory and the anchoring drift (room held at target by the base setpoint)
        s_ref = np.empty(self.n + 1)
        s_ref[0] = self.f_ref[0]
        for i in range(self.n):
            s_ref[i + 1] = s_ref[i] + self.a * (self.f_ref[i] - s_ref[i])
        self.s_ref = s_ref
        r = cfg.target
        self.drift = -(self.ks * (s_ref[:-1] - r) - self.ko * (r - t_out))

    # --- pump -------------------------------------------------------------------------------
    def flow_pump(self, u):
        p = self.pump
        return p.flow_intercept + p.flow_slope * self.t + p.flow_per_rts * (np.asarray(u) - p.rts_ref)

    def energy(self, u):
        """Compressor + electric heater kWh per hour for setpoint path u (never above 3.6 kW compressor)."""
        p = self.pump
        raw = p.energy_intercept + p.energy_per_outdoor * self.t + p.energy_per_rts * u
        comp = np.clip(raw, 0.0, COMPRESSOR_MAX_KWH)
        aux = p.aux_per_flow_c * np.maximum(self.flow_pump(u) - AUX_ONSET, 0.0)
        return comp, aux

    # --- non-linear replay ------------------------------------------------------------------
    def _flow_i(self, u, i):
        p = self.pump
        return p.flow_intercept + p.flow_slope * self.t[i] + p.flow_per_rts * (u - p.rts_ref)

    def step(self, i, r, s, u):
        """One hour of the room/slab model; returns (room, slab) at the end of hour i."""
        c = self.cfg
        flow = min(self._flow_i(u, i), c.flow_cap)
        valve = min(max((c.band_hi + 0.5 - r) / 0.5, 0.0), 1.0)
        r_next = r + valve * self.ks * (s - r) - self.ko * (r - self.t[i]) + self.drift[i]
        s_next = s + (valve * flow + (1 - valve) * r - s) * self.a
        return r_next, s_next

    def simulate(self, u, x0=0.0, y0=0.0):
        """Replay a setpoint path. Returns dict of arrays (room at end of each hour etc.)."""
        n = len(u)
        R = np.empty(n)
        r, s = self.cfg.target + x0, self.s_ref[0] + y0
        for i in range(n):
            r, s = self.step(i, r, s, u[i])
            R[i] = r
        comp, aux = self.energy(np.asarray(u, dtype=float))
        return {"room": R, "comp": comp, "aux": aux, "kwh": comp + aux, "cost": self.price * (comp + aux)}

    def slice(self, i0, i1):
        """The same world restricted to hours [i0, i1) (shares the global reference path)."""
        v = copy.copy(self)
        for name in ("hours", "price", "t", "sun", "u_cap", "base_eff", "u_max", "u_min",
                     "flow_pump_ref", "f_ref", "drift"):
            setattr(v, name, getattr(self, name)[i0:i1])
        v.s_ref = self.s_ref[i0:i1 + 1]
        v.n = i1 - i0
        v.i0 = self.i0 + i0
        return v

    def kwh_per_c_hour(self):
        """Electricity (kWh/h) it takes to hold the room one degree warmer in the long run."""
        p = self.pump
        gain = self.ks / (self.ks + self.ko)
        return p.energy_per_rts / (gain * p.flow_per_rts)

    def kwh_per_c_stock(self):
        """Electricity (kWh) that one degree of stored heat is worth (hold-cost over the time constant)."""
        return self.kwh_per_c_hour() / (self.ks + self.ko)

    def baseline(self):
        return self.simulate(self.base_eff.copy())


def optimise(w: World, price=None, terminal_value: float = 0.0, comfort_weight: float = 0.0,
             x0: float = 0.0, y0: float = 0.0, u_prev: float | None = None,
             end_floor: bool = True, neutral: bool | None = None, cold_weight: float = 0.0,
             band_margin: float = 0.0, exec_sums: dict | None = None,
             block_penalty: float | None = None, end_deficit_cost: float = 0.0):
    """Minimise heating cost over the whole window of `w` (LP). Returns the setpoint path u.

    `cold_weight` (SEK per C*h) prices every degree-hour the room spends below target.

    `price` is what the optimiser believes (default: true price). `terminal_value` credits the
    heat left in the room at the end (SEK per C). `comfort_weight` penalises degree-hours below
    target (SEK per C*h) -- the Pareto weight; `neutral` (config) forces the mean to target.
    """
    c, p, n = w.cfg, w.pump, w.n
    true_price = w.price if price is None else price
    price = np.maximum(true_price, 0.0)
    raw_ref = p.energy_intercept + p.energy_per_outdoor * w.t + p.energy_per_rts * w.base_eff
    # variable layout: du, f, x, y, e, aux, shortfall
    iu, i_f, ix, iy, ie, ia, isf = (np.arange(n) + k * n for k in range(7))
    nblk = 0
    if (c.neutral if neutral is None else neutral) and block_penalty is not None:
        blk_len = (w.n + w.i0) if c.neutral_days == 0 else 24 * c.neutral_days
        nblk = (w.i0 + n - 1) // blk_len - w.i0 // blk_len + 1
    nv = 7 * n + nblk
    cost = np.zeros(nv)
    raw_lo = (p.energy_intercept + p.energy_per_outdoor * w.t + p.energy_per_rts * w.base_eff
              + p.energy_per_rts * (w.u_min - w.base_eff))
    # a negative price pays for consuming: only valid where the compressor draws power at every setpoint
    paid = (true_price < 0) & (raw_lo > 0)
    cost[ie] = np.where(paid, true_price, price)
    cost[ia] = price
    cost[i_f] = -1e-4
    cost[ix] = -comfort_weight  # reward each degree-hour above target (linear, symmetric)
    cost[isf] = cold_weight
    if nblk:
        cost[7 * n:] = block_penalty
    cost[isf[-1]] += end_deficit_cost  # one-sided: ending colder than target is billed
    cost[ix[-1]] += -terminal_value
    lb, ub = np.full(nv, -np.inf), np.full(nv, np.inf)
    lb[iu], ub[iu] = w.u_min - w.base_eff, w.u_max - w.base_eff
    lb[i_f], ub[i_f] = -60.0, np.maximum(c.flow_cap - w.f_ref, 0.0)
    lb[iy], ub[iy] = -80.0, 80.0
    lb[ix], ub[ix] = c.band_lo - c.target - band_margin, c.band_hi - c.target + band_margin
    lb[ie], lb[ia] = 0.0, 0.0
    lb[isf], ub[isf] = 0.0, 20.0
    if nblk:
        lb[7 * n:], ub[7 * n:] = 0.0, 1e5
    ub[ie], ub[ia] = COMPRESSOR_MAX_KWH, 20.0
    # equalities: x_t = (1-ks-ko) x_{t-1} + ks y_{t-1};  y_t = (1-a) y_{t-1} + a f_t
    rows, cols, vals = [], [], []
    b_eq = np.zeros(2 * n)
    def add(r, cc, v):
        rows.append(r); cols.append(cc); vals.append(v)
    for t in range(n):
        add(t, ix[t], 1.0)
        if t > 0:
            add(t, ix[t - 1], -(1 - w.ks - w.ko)); add(t, iy[t - 1], -w.ks)
        else:
            b_eq[t] = (1 - w.ks - w.ko) * x0 + w.ks * y0
        r2 = n + t
        add(r2, iy[t], 1.0); add(r2, i_f[t], -w.a)
        if t > 0:
            add(r2, iy[t - 1], -(1 - w.a))
        else:
            b_eq[r2] = (1 - w.a) * y0
    idx_paid = np.where(paid)[0]  # negative price: consumption equals what the setpoint draws
    for j, t in enumerate(idx_paid):
        add(2 * n + j, ie[t], 1.0); add(2 * n + j, iu[t], -p.energy_per_rts)
    b_eq = np.concatenate([b_eq, raw_ref[idx_paid]])
    A_eq = sp.csr_matrix((vals, (rows, cols)), shape=(2 * n + len(idx_paid), nv))
    # inequalities (A_ub z <= b_ub)
    r_, c_, v_, b = [], [], [], []
    k = 0
    def addu(cc, v):
        r_.append(k); c_.append(cc); v_.append(v)
    if end_deficit_cost > 0:  # shortfall at the last hour >= -x_end
        addu(isf[-1], -1.0); addu(ix[-1], -1.0); b.append(0.0); k += 1
    if cold_weight > 0:
        for t in range(n):  # shortfall_t >= -x_t
            addu(isf[t], -1.0); addu(ix[t], -1.0); b.append(0.0); k += 1
    for t in range(n):  # f_t <= flow_per_rts * du_t + (pump flow above the shunt cap at the base setpoint)
        addu(i_f[t], 1.0); addu(iu[t], -p.flow_per_rts); b.append(w.flow_pump_ref[t] - w.f_ref[t]); k += 1
    for t in range(n):  # e_t >= raw_ref + e_per_rts du_t (paid hours: equality, below)
        if paid[t]:
            continue
        addu(ie[t], -1.0); addu(iu[t], p.energy_per_rts); b.append(-raw_ref[t]); k += 1
    for t in range(n):  # aux_t >= aux_per_flow (flow_ref + fps du - onset)
        addu(ia[t], -1.0); addu(iu[t], p.aux_per_flow_c * p.flow_per_rts)
        b.append(-p.aux_per_flow_c * (w.flow_pump_ref[t] - AUX_ONSET)); k += 1
    if c.neutral if neutral is None else neutral:  # every block of neutral_days: mean x >= 0
        block = (w.n + w.i0) if c.neutral_days == 0 else 24 * c.neutral_days
        blocks = {}
        for t in range(n):
            blocks.setdefault((w.i0 + t) // block, []).append(t)
        for blk, ts in blocks.items():
            for t in ts:
                addu(ix[t], -1.0)
            if nblk:  # soft: a deficit variable priced at block_penalty per C*h
                addu(7 * n + (blk - w.i0 // block), -1.0)
            # hours of this block already executed count towards it (rolling plans)
            done = 0.0 if exec_sums is None else exec_sums.get(blk, 0.0)
            b.append(NEUTRAL_SLACK_CH + done); k += 1
    if c.max_step is not None:
        for t in range(1, n):
            addu(iu[t], 1.0); addu(iu[t - 1], -1.0)
            b.append(c.max_step - (w.base_eff[t] - w.base_eff[t - 1])); k += 1
        if u_prev is not None:
            addu(iu[0], 1.0); b.append(c.max_step + u_prev - w.base_eff[0]); k += 1
    if c.max_step_down is not None:
        for t in range(1, n):
            addu(iu[t], -1.0); addu(iu[t - 1], 1.0)
            b.append(c.max_step_down + (w.base_eff[t] - w.base_eff[t - 1])); k += 1
    A_ub = sp.csr_matrix((v_, (r_, c_)), shape=(k, nv)); b_ub = np.array(b)
    if end_floor:  # do not end below the reference level: no borrowing from the future
        lb[ix[-1]] = max(lb[ix[-1]], -END_SLACK_C)
    bounds = np.column_stack([lb, ub])
    # HiGHS' interior point is sensitive to the cost scale on some of these degenerate problems
    for scale in (1.0, 0.1, 100.0, 0.01, 10.0):
        res = linprog(cost * scale, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
                      method="highs-ipm", options={"presolve": False})
        if res.status == 0:
            res.fun = res.fun / scale
            break
    if res.status != 0:
        raise RuntimeError(f"LP failed: status {res.status} {res.message}")
    return w.base_eff + res.x[iu], res


# --- data -------------------------------------------------------------------------------------
def load_study():
    d = np.load(SCRATCH / "study.npz")
    hours = np.array([dt.datetime.fromtimestamp(x, dt.UTC) for x in d["hours"]])
    return hours, {k: d[k] for k in d.files if k != "hours"}


def winters():
    """(label, start, stop) UTC windows: 1 Nov -> 15 Apr local, four winters."""
    out = []
    for y in (2022, 2023, 2024, 2025):
        a = dt.datetime(y, 11, 1, tzinfo=TZ).astimezone(dt.UTC)
        b = dt.datetime(y + 1, 4, 16, tzinfo=TZ).astimezone(dt.UTC)
        out.append((f"{y}/{str(y + 1)[2:]}", a, b))
    return out


def window(hours, series, a, b):
    m = (hours >= a) & (hours < b)
    return hours[m], {k: v[m] for k, v in series.items()}


def local_day_index(hours):
    """Integer local-calendar-day index per hour and the day list."""
    days = np.array([h.astimezone(TZ).date().toordinal() for h in hours])
    return days - days.min(), days
