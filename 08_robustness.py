"""Robustness of the empirical slot grid and of the slot-error counterfactual.

T1 free spacing fit: pooled circular coherence R(s) of in-plane phases, s scanned 4-25 deg
T2 phase estimators: circular mean / circular median / Huber-weighted mean
T3 leave-out: drop each satellite of the closest pairs (and all of them) from the grid estimate
T4 epochs: grid coherence and per-satellite slot-error persistence on several dates
T5 bootstrap: resample satellites within each plane

The tracked quantity is d_slotted: the model minimum distance of the closest pairs after removing
their slot errors, plus the fraction of satellites within 1 deg of the grid.

Usage:  py 08_robustness.py
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from common import DATA, OUT, load_gp, load_states

_spec = importlib.util.spec_from_file_location("cm", OUT.parent / "07_crossing_model.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)
pl = cm.planes_mod

S = 10.0
DATES = ["2025-06-01", "2025-12-01", "2026-04-01", "2026-07-15", "2026-09-22"]


def wrap_s(x, s=S):
    return (x + s / 2) % s - s / 2


def get_el(df, states, t):
    near = df.assign(_dt=(df["EPOCH"] - t).abs())
    near = near[near["_dt"] <= pd.Timedelta(hours=48)]
    near = near.loc[near.groupby("NORAD_CAT_ID")["_dt"].idxmin()]
    near = near.merge(states[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"])
    op = near[near["STATE"] == "operational"].reset_index(drop=True)
    el = pl.elements_at(op, t)
    el["plane"] = pl.cluster_planes(el["raan"].to_numpy(), 2.0)
    el = el.join(op[["A_KM", "ECCENTRICITY", "ARG_OF_PERICENTER"]])
    el["ub"] = cm.mean_phase(el["u"], el["ECCENTRICITY"], el["ARG_OF_PERICENTER"])   # mean argument of latitude
    return el


def circ_median(u, s=S):
    grid = np.arange(0, s, 0.01)
    cost = np.abs(wrap_s(u[None, :] - grid[:, None], s)).sum(1)
    return float(grid[cost.argmin()])


def huber_mean(u, s=S, c=0.3):
    ph = pl.circmean(u, s)
    for _ in range(30):
        e = wrap_s(u - ph, s)
        w = 1 / np.maximum(1, np.abs(e) / c)
        ph = (ph + (w * e).sum() / w.sum()) % s
    return float(ph)


ESTIMATORS = {"circular mean": lambda u: pl.circmean(u, S), "circular median": circ_median, "Huber (c=0.3)": huber_mean}


PHASE_COL = "um"      # slot errors in SGP4 mean argument of latitude (see manuscript Sections 5.4, 6.0)


def slots(el, est, exclude=(), rng=None, col=None):
    """Slot error from the SGP4 mean argument of latitude ("um"; falls back to "ub" if absent, or col= to force one).

    ub_slotted is the equation-of-centre phase used by the crossing model minus that slot error, i.e. the satellite moved
    onto its slot with every other element kept. The -0.114 deg cos u difference between ub and um cancels at the polar
    crossing but not between satellites of one plane, so slot errors are measured in um."""
    col = col or (PHASE_COL if PHASE_COL in el else "ub")
    phase = {}
    for p, g in el.groupby("plane"):
        g = g[~g["OBJECT_ID"].isin(exclude)]
        u = (g[col].to_numpy() % S)
        if rng is not None:
            u = rng.choice(u, len(u))
        phase[p] = est(u) if len(u) else np.nan
    e = el.copy()
    e["slot_err_deg"] = wrap_s(e[col] - e["plane"].map(phase))
    e["ub_slotted"] = e["ub"] - e["slot_err_deg"]
    return e


def pair_index(el, pairs):
    idx = {o: i for i, o in enumerate(el["OBJECT_ID"])}
    return (np.array([idx[a] for a, _ in pairs]), np.array([idx[b] for _, b in pairs]))


def dslot(el_s, iu, ju):
    return cm.model_mean(el_s, iu, ju, "ub_slotted")["d_model_km"].to_numpy()


def coherence(el, s):
    R, n = [], []
    for _, g in el.groupby("plane"):
        if len(g) >= 5:
            R.append(abs(np.exp(2j * np.pi * g[PHASE_COL if PHASE_COL in g else "ub"].to_numpy() / s).mean())), n.append(len(g))
    return float(np.average(R, weights=n))


def main():
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    t0 = df["EPOCH"].max().floor("h") - pd.Timedelta(days=1)
    el = get_el(df, states, t0)
    tab = pd.read_csv(OUT / "crossing_model_pairs.csv").nsmallest(5, "observed_dmin_km")
    pairs = list(zip(tab["id_1"], tab["id_2"]))
    iu, ju = pair_index(el, pairs)
    base = dslot(slots(el, ESTIMATORS["circular mean"]), iu, ju)
    label = [f"{a}/{b}" for a, b in pairs]
    print(f"epoch {t0:%Y-%m-%d}; five closest pairs (observed {tab['observed_dmin_km'].round(2).tolist()} km)")
    print("baseline d_slotted:", dict(zip(label, base.round(1))), "\n")

    print("T1  pooled in-plane coherence R(s) (1 = all satellites on a grid of spacing s)")
    grid = np.arange(4, 25.01, 0.05)
    R = np.array([coherence(el, s) for s in grid])
    for s in (5, 8, 9, 9.5, 10, 10.5, 11, 15, 20):
        print(f"   R({s:>4}) = {R[np.argmin(abs(grid - s))]:.3f}")
    sel = (grid >= 8) & (grid <= 25)
    strong = grid[sel][R[sel] >= 0.9]
    print(f"   largest s in 8-25 deg with R >= 0.90: {strong.max() if len(strong) else 'none'}   "
          f"(range with R>=0.9: {strong.min():.2f}-{strong.max():.2f})" if len(strong) else "   none >= 0.9")
    print(f"   satellites within 1 deg of the 10-deg grid: {(slots(el, ESTIMATORS['circular mean'])['slot_err_deg'].abs() <= 1).mean():.1%}\n")

    print("T2  phase estimators -> d_slotted (km) of the five closest pairs")
    rows = {name: dslot(slots(el, fn), iu, ju) for name, fn in ESTIMATORS.items()}
    print(pd.DataFrame(rows, index=label).round(1).to_string(), "\n")

    print("T3  leave-out: grid re-estimated without the named satellite(s)")
    involved = sorted({o for p in pairs for o in p})
    out = np.array([dslot(slots(el, ESTIMATORS["circular mean"], exclude={o}), iu, ju) for o in involved])
    allout = dslot(slots(el, ESTIMATORS["circular mean"], exclude=set(involved)), iu, ju)
    print(pd.DataFrame({"baseline": base, "min over LOO": out.min(0), "max over LOO": out.max(0),
                        "all 10 removed": allout}, index=label).round(1).to_string(), "\n")

    print("T4  epochs: grid coherence at s=10 and slot-error persistence")
    prev = None
    for d in DATES:
        e = slots(get_el(df, states, pd.Timestamp(d, tz="UTC")), ESTIMATORS["circular mean"])
        line = (f"   {d}: n={len(e):3d} planes={e['plane'].nunique():2d}  R(10)={coherence(e, 10):.3f}  "
                f"median |slot err|={e['slot_err_deg'].abs().median():.2f} deg  within 1 deg={(e['slot_err_deg'].abs() <= 1).mean():.0%}")
        if prev is not None:
            m = e[["OBJECT_ID", "slot_err_deg"]].merge(prev[["OBJECT_ID", "slot_err_deg"]], on="OBJECT_ID")
            if len(m) > 20:
                line += (f"  | same satellites vs previous date (n={len(m)}): r={np.corrcoef(m['slot_err_deg_x'], m['slot_err_deg_y'])[0, 1]:.2f}, "
                         f"median |change|={(m['slot_err_deg_x'] - m['slot_err_deg_y']).abs().median():.2f} deg")
        print(line)
        prev = e
    print()

    print("T5  bootstrap of the grid (500 resamples of satellites within each plane)")
    rng = np.random.default_rng(0)
    B = np.array([dslot(slots(el, ESTIMATORS["circular mean"], rng=rng), iu, ju) for _ in range(500)])
    print(pd.DataFrame({"baseline": base, "p5": np.percentile(B, 5, 0), "median": np.median(B, 0),
                        "p95": np.percentile(B, 95, 0)}, index=label).round(1).to_string())


if __name__ == "__main__":
    main()
