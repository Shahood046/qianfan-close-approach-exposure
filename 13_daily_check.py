"""Does the 3-day sampling bias the main results? Compare a daily screen with the 3-day screen on the same period.

Inputs (run first):
    py 04_screen.py --every-days 1 --start 2026-04-01 --end 2026-09-22 --kinds QQ --tag _dailyA
    py 11_per_angle.py --pairs encounter_pairs_dailyA.csv --tag _dailyA
and the 3-day outputs (encounter_pairs_3d.csv, data/pair_windows.pkl.gz).

Checks:
  1 the daily windows that coincide with 3-day windows reproduce the 3-day screen exactly
  2 rates (on-grid, slowly drifting, standard angles): daily vs 3-day windows in the same period
  3 even/odd plane-step rates
  4 frequency by nominal margin and the concentration curve
  5 episodes: durations at daily resolution, and how many episodes the 3-day cadence would miss

Usage:  py 13_daily_check.py [--since 2026-04-01]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2

from common import DATA, OUT


def ci(k, n):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return 1000 * lo / n, 1000 * chi2.ppf(0.975, 2 * k + 2) / 2 / n


def episodes(ev: pd.DataFrame, windows: list, gap: int = 1) -> pd.DataFrame:
    """Consecutive-window runs (allowing `gap` missed windows) of the same pair among event pair-windows."""
    idx = {t: i for i, t in enumerate(windows)}
    ev = ev.assign(w=ev["t0"].map(idx))
    rows = []
    for (a, b), d in ev.sort_values("w").groupby(["a", "b"]):
        run = []
        for _, r in d.iterrows():
            if run and r["w"] - run[-1]["w"] > gap + 1:
                rows.append(run)
                run = []
            run.append(r)
        if run:
            rows.append(run)
    return pd.DataFrame([{"a": r[0]["a"], "b": r[0]["b"], "start": r[0]["t0"], "end": r[-1]["t0"], "n_win": len(r),
                          "days": [x["t0"] for x in r], "dmin": min(x["d_obs"] for x in r),
                          "near60": any(x["nominal"] < 60 for x in r)} for r in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default="2026-04-01")
    args = ap.parse_args()
    since = pd.Timestamp(args.since, tz="UTC")

    D = pd.read_pickle(DATA / "pair_windows_dailyA.pkl.gz")
    T = pd.read_pickle(DATA / "pair_windows.pkl.gz")
    for X in (D, T):
        X["t0"] = pd.to_datetime(X["t0"], utc=True)
    T = T[T["t0"] >= since]
    dw, tw = sorted(D["t0"].unique()), sorted(T["t0"].unique())
    print(f"daily windows {len(dw)}, 3-day windows {len(tw)}; coincident: {len(set(dw) & set(tw))}")

    # 1 exact reproduction on coincident windows
    Dc = D[D["t0"].isin(set(tw))]
    m = Dc.merge(T, on=["t0", "a", "b"], suffixes=("_d", "_t"))
    same = ((m["obs10_d"] == m["obs10_t"]) & (np.isclose(m["nominal_d"], m["nominal_t"]))).mean()
    print(f"1. coincident windows: {len(m):,} pair-windows matched, identical event flag and nominal margin: {same:.4%}")

    good = lambda X: X[X["good"] & X["std_angle"]]
    Dg, Tg = good(D), good(T)
    print("\n2. rate of <= 10 km per 1000 on-grid, slow, standard-angle pair-windows, same period")
    for name, X in (("daily windows", Dg), ("3-day windows", Tg)):
        k, n = int(X["obs10"].sum()), len(X)
        lo, hi = ci(k, n)
        print(f"   {name}: {k} events / {n:,} pair-windows = {1000 * k / n:.3f}  (95% CI {lo:.3f}-{hi:.3f})")

    print("\n3. even vs odd plane steps")
    for name, X in (("daily", Dg), ("3-day", Tg)):
        ev = X["k"] % 2 == 0
        e, o = 1000 * X.loc[ev, "obs10"].mean(), 1000 * X.loc[~ev, "obs10"].mean()
        print(f"   {name}: even {e:.3f}, odd {o:.3f} per 1000 pair-windows; ratio {e / o:.0f}")
    print("   by k, daily vs 3-day (per 1000):")
    kk = pd.DataFrame({"daily": 1000 * Dg.groupby("k")["obs10"].mean(), "3-day": 1000 * Tg.groupby("k")["obs10"].mean()}).round(3)
    print(kk.to_string())

    print("\n4. frequency by nominal margin (per 1000 pair-windows within 10 km) and concentration")
    bins = [0, 10, 30, 60, 100, 1e9]
    rows = {}
    for name, X in (("daily", Dg), ("3-day", Tg)):
        b = pd.cut(X["nominal"], bins)
        rows[name] = (X.groupby(b, observed=True)["obs10"].mean() * 1000).round(2)
    print(pd.DataFrame(rows).to_string())
    for name, X in (("daily", Dg), ("3-day", Tg)):
        near = X["nominal"] < 60
        print(f"   {name}: nominal<60 km: {near.mean():.2%} of pair-windows carry {(X['obs10'] & near).sum()}/{int(X['obs10'].sum())} = "
              f"{(X['obs10'] & near).sum() / X['obs10'].sum():.1%} of events")

    print("\n5. episodes (event pair-windows in the on-grid, slow, standard-angle subset; gap <= 1 window)")
    ev_d = Dg[Dg["obs10"]]
    ep_d = episodes(ev_d, dw)
    ep_t = episodes(Tg[Tg["obs10"]], tw)
    step = pd.Series(dw).diff().median()
    print(f"   daily: {len(ep_d)} episodes, duration (days, {step.days}-day resolution) median {ep_d['n_win'].median():.0f}, "
          f"90th pct {ep_d['n_win'].quantile(.9):.0f}, max {ep_d['n_win'].max()}; single-day episodes: {(ep_d['n_win'] == 1).mean():.0%}")
    print(f"   3-day: {len(ep_t)} episodes (3-day windows, gap <= 1 window)")
    tset = set(tw)
    seen = ep_d["days"].apply(lambda ds: any(d in tset for d in ds))
    print(f"   of the {len(ep_d)} daily episodes, {seen.sum()} ({seen.mean():.0%}) contain a day that a 3-day cadence would have sampled; "
          f"{(~seen).sum()} ({(~seen).mean():.0%}) would be missed entirely")
    if (~seen).any():
        missed = ep_d[~seen]
        print(f"   missed episodes: median duration {missed['n_win'].median():.0f} d, median d_min {missed['dmin'].median():.1f} km, "
              f"share from nominal<60 pairs {missed['near60'].mean():.0%}")
    print(f"   episodes from nominal<60 pairs: daily {ep_d['near60'].mean():.1%}, 3-day {ep_t['near60'].mean():.1%}")
    short = ep_d[seen]
    print(f"   durations of detected episodes: daily median {short['n_win'].median():.0f} d; 3-day-cadence medians of "
          f"the same episodes would read {np.ceil(short['n_win'] / 3).median():.0f} windows of 3 days")
    ep_d.drop(columns="days").to_csv(OUT / "episodes_dailyA.csv", index=False)


if __name__ == "__main__":
    main()
