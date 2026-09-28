"""Sensitivity of the main quantitative conclusions to the definition of the 'on-grid, slowly drifting' subset.

For slot-error limits {0.5, 1, 1.5, 2} deg and drift limits {20, 50, 100} km/day (standard crossing angles only) it reports:
  - even/odd plane-step rates (pairs within 10 km per 1000 pair-windows) and their ratio
  - agreement of the model with the SGP4 screen (precision / recall of model window-min <= 10 km)
  - concentration: share of events in pair-windows with nominal margin < 60 km (and the share of pair-windows)
  - frequency by nominal margin bin
  - counterfactual at event level: for events with observed d_min < 3 km, the nominal (slot errors removed) distance

Input: data/pair_windows{tag}.pkl.gz written by 11_per_angle.py.

Usage:  py 12_sensitivity.py [--tag ""] [--since 2024-01-01]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import DATA


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="")
    ap.add_argument("--since", default=None)
    args = ap.parse_args()
    P = pd.read_pickle(DATA / f"pair_windows{args.tag}.pkl.gz")
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    if args.since:
        P = P[P["t0"] >= pd.Timestamp(args.since, tz="UTC")]
    P = P[P["std_angle"]]
    P["even"] = P["k"] % 2 == 0
    print(f"{len(P):,} pair-windows at standard angles, {P['t0'].nunique()} windows\n")

    rows = []
    for sm in (0.5, 1.0, 1.5, 2.0):
        for dr in (20, 50, 100):
            D = P[(P["slot_max"] <= sm) & (P["drift"] < dr)]
            ev = D["obs10"]
            near = D["nominal"] < 60
            rate_e = 1000 * ev[D["even"]].mean()
            rate_o = 1000 * ev[~D["even"]].mean()
            tp, fp, fn = int((D["mod10"] & ev).sum()), int((D["mod10"] & ~ev).sum()), int((~D["mod10"] & ev).sum())
            e3 = D[ev & (D["d_obs"] < 3)]
            rows.append({"slot<=": sm, "drift<": dr, "pair_windows": len(D), "events": int(ev.sum()),
                         "even_per_1000": rate_e, "odd_per_1000": rate_o, "even/odd": rate_e / rate_o if rate_o else np.inf,
                         "model_recall": tp / (tp + fn) if tp + fn else np.nan, "model_precision": tp / (tp + fp) if tp + fp else np.nan,
                         "share_events_near60": (ev & near).sum() / max(ev.sum(), 1), "share_pw_near60": near.mean(),
                         "cf_events<3km": len(e3), "cf_median_nominal": e3["nominal"].median(),
                         "cf_share>10": (e3["nominal"] > 10).mean(), "cf_share>20": (e3["nominal"] > 20).mean()})
    R = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))

    print("\nFrequency by nominal-margin bin (per 1000 pair-windows within 10 km), by subset")
    bins = [0, 10, 30, 60, 1e9]
    out = []
    for sm in (0.5, 1.0, 2.0):
        for dr in (20, 50, 100):
            D = P[(P["slot_max"] <= sm) & (P["drift"] < dr)]
            b = pd.cut(D["nominal"], bins)
            r = D.groupby(b, observed=True)["obs10"].mean() * 1000
            out.append({"slot<=": sm, "drift<": dr, **{str(k): v for k, v in r.items()}})
    print(pd.DataFrame(out).round(2).to_string(index=False))

    print("\nRanges across all 12 subsets:")
    for c in ("even/odd", "model_recall", "model_precision", "share_events_near60", "share_pw_near60", "cf_median_nominal", "cf_share>10", "cf_share>20"):
        v = R[c].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {c:22s} min {v.min():.3f}  max {v.max():.3f}")
    R.to_csv(DATA.parent / "out" / f"sensitivity{args.tag}.csv", index=False)


if __name__ == "__main__":
    main()
