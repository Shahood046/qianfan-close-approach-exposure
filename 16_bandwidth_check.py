"""Sensitivity of the locked results to the arrival band width X (K = 30 d fixed).

Prerequisites (baseline X = 15 km already exists as data/pair_windows_daily.pkl.gz):
    py 02_states.py --X 10 --tag _X10
    py 02_states.py --X 30 --tag _X30
    py 11_per_angle.py --pairs encounter_pairs_daily.csv --tag _dailyX10 --states-tag _X10
    py 11_per_angle.py --pairs encounter_pairs_daily.csv --tag _dailyX30 --states-tag _X30
The SGP4 screen covered every Qianfan satellite whatever its state, so only the labels change; 11_per_angle
relabels the screened pairs with the element set the screen used (reproduces the baseline labels exactly).

Writes BANDWIDTH_CHECK.md. Every statistic uses the same code as 14_locked_claims.py.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from common import DATA, OUT, ROOT

_spec = importlib.util.spec_from_file_location("lc", ROOT / "14_locked_claims.py")
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

SETTINGS = [("X = 10 km", "_dailyX10", "_X10"), ("X = 15 km (baseline)", "_daily", ""), ("X = 30 km", "_dailyX30", "_X30")]
DATES = ["2025-06-01", "2026-01-01", "2026-06-01", "2026-09-22"]


def stats(P: pd.DataFrame, sats: pd.DataFrame) -> dict:
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    windows = sorted(P["t0"].unique())
    G = P[P["good"] & P["std_angle"]]
    s = {"Arrived satellites (whole record)": int((sats["status"] == "arrived").sum()),
         "Stranded satellites": int((sats["status"] == "stranded").sum())}
    nop = P.groupby("t0")["n_op"].first()
    s["Operational satellites on " + ", ".join(d[2:] for d in DATES)] = " / ".join(
        str(int(nop.iloc[nop.index.get_indexer([pd.Timestamp(d, tz="UTC")], method="nearest")[0]])) for d in DATES)
    s["Daily windows"] = len(windows)
    s["Operational inter-plane pair-windows"] = f"{len(P):,}"
    s["Primary-population pair-windows"] = f"{len(G):,} ({len(G) / len(P):.1%})"
    k, n = int(G["obs10"].sum()), len(G)
    lo, hi = lc.poisson_ci(k, n)
    s["Events <= 10 km (primary)"] = k
    s["Event rate per 1000 (primary)"] = f"{1000 * k / n:.3f} ({lo:.3f}-{hi:.3f})"
    early = G["t0"] < lc.EARLY_END
    s["Event rate early / recent"] = f"{1000 * G[early]['obs10'].mean():.3f} / {1000 * G[~early]['obs10'].mean():.3f}"
    U = P[P["std_angle"]]
    s["Event rate, no on-grid/drift restriction"] = f"{1000 * U['obs10'].mean():.3f} ({int(U['obs10'].sum())} events)"
    fr = np.array([0.005, 0.01, 0.015])
    pt = lc.share_at(G["nominal"].to_numpy(), G["obs10"].to_numpy(), fr)
    sh, _ = lc.block_boot(G, fr)
    b = np.percentile(sh[:, 1], [2.5, 97.5])
    s["1% concentration (95% block bootstrap)"] = f"{pt[1]:.1%} ({b[0]:.1%}-{b[1]:.1%})"
    s["1.5% concentration"] = f"{pt[2]:.1%}"
    s["0.5% concentration"] = f"{pt[0]:.1%}"
    per = []
    for name, X in (("early", G[early]), ("recent", G[~early])):
        per.append(f"{lc.share_at(X['nominal'].to_numpy(), X['obs10'].to_numpy(), [0.01])[0]:.1%}")
    s["1% concentration early / recent"] = " / ".join(per)
    ev_e = 1000 * G[G["k"] % 2 == 0]["obs10"].mean()
    ev_o = 1000 * G[G["k"] % 2 == 1]["obs10"].mean()
    s["Even / odd rate per 1000 (ratio)"] = f"{ev_e:.2f} / {ev_o:.3f} ({ev_e / ev_o:.0f})"
    g2 = G[G["mod10"] | G["obs10"]]
    tp, fp, fn = int((g2["mod10"] & g2["obs10"]).sum()), int((g2["mod10"] & ~g2["obs10"]).sum()), int((~g2["mod10"] & g2["obs10"]).sum())
    s["Model recall / precision"] = f"{tp / (tp + fn):.1%} / {tp / (tp + fp):.1%}"
    ep = lc.episodes(G[G["obs10"]], windows)
    s["Daily episodes (distinct pairs)"] = f"{len(ep)} ({ep.groupby(['a', 'b']).ngroups})"
    s["Episode duration median / P90 / max (d)"] = f"{ep['days'].median():.0f} / {ep['days'].quantile(.9):.0f} / {ep['days'].max()}"
    s["One-day episodes"] = f"{(ep['days'] == 1).mean():.0%}"
    long_ep = ep[ep["days"] >= 3]
    s["Episodes >= 3 d with nominal margin < 100 km"] = f"{(long_ep['min_nominal'] < 100).mean():.1%} of {len(long_ep)}"
    close = ep[ep["dmin"] < 3]
    s["Counterfactual, episodes with d_min < 3 km: n; nominal median; share > 10 km; > 20 km"] = (
        f"{len(close)}; {close['nominal_at_min'].median():.1f} km; {(close['nominal_at_min'] > 10).mean():.0%}; "
        f"{(close['nominal_at_min'] > 20).mean():.0%}")
    s["Episodes remaining < 5 km after removing slot errors"] = f"{(ep['nominal_at_min'] < 5).mean():.0%}"
    return s


def main():
    cols, sets = {}, {}
    for name, ptag, stag in SETTINGS:
        P = pd.read_pickle(DATA / f"pair_windows{ptag}.pkl.gz")
        sats = pd.read_csv(OUT / f"satellite_summary{stag}.csv")
        print(f"--- {name}", flush=True)
        cols[name] = stats(P, sats)
        G = P[P["good"] & P["std_angle"]]
        sets[name] = (set(zip(P["t0"], P["a"], P["b"])), set(zip(G["t0"], G["a"], G["b"])),
                      set(zip(G.loc[G["obs10"], "t0"], G.loc[G["obs10"], "a"], G.loc[G["obs10"], "b"])))
    T = pd.DataFrame(cols)
    base = "X = 15 km (baseline)"
    diff = []
    for name in cols:
        if name == base:
            continue
        a, g, e = sets[name]
        A, Gb, E = sets[base]
        diff.append(f"- {name} vs baseline: all pair-windows +{len(a - A):,} / -{len(A - a):,}; primary pair-windows "
                    f"+{len(g - Gb):,} / -{len(Gb - g):,}; primary events +{len(e - E)} / -{len(E - e)}")
    lines = ["# Band-width sensitivity (auto-generated by 16_bandwidth_check.py)", "",
             "Arrival band half-width X varied with K = 30 d; everything else identical (same SGP4 screen, daily windows, "
             "subset definitions, estimators). Primary population = standard angle, on-grid (<= 1 deg), relative drift < 50 km/day.", "",
             "| Quantity | " + " | ".join(T.columns) + " |", "|---|" + "---|" * len(T.columns)]
    lines += [f"| {q} | " + " | ".join(str(v) for v in r) + " |" for q, r in T.iterrows()]
    lines += ["", "Set differences (pair-window keys t0, a, b):"] + diff
    (ROOT / "BANDWIDTH_CHECK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
