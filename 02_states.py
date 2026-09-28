"""Classify each Qianfan element set into an orbital-transition state and check the go/no-go criteria.

States (defined retrospectively from each satellite's own altitude history, no design assumptions):
    drift        before arrival, altitude not rising (parking at insertion altitude)
    ascent       before arrival, altitude rising faster than --slope km/day
    operational  from the arrival epoch on: first time the altitude enters |alt - op_alt| <= X
                 and stays there (>= 95% of element sets) for K days
    excursion    after arrival but outside the band
    stranded     never arrived, launched >= --stranded-age days before the data cutoff,
                 and flat over the final 60 days below the operational band

op_alt is estimated from the data (the mode of the satellites' terminal altitudes).
X and K are swept to show how much the arrival epochs depend on them.

Usage:  py 02_states.py [--X 15] [--K 30] [--slope 1.0]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, load_gp

SWEEP_X = [10, 15, 20, 30]
SWEEP_K = [14, 30]
COLORS = {"drift": "tab:orange", "ascent": "tab:red", "operational": "tab:blue",
          "excursion": "tab:purple", "stranded": "black"}


def operational_altitude(df: pd.DataFrame) -> float:
    last = df.groupby("NORAD_CAT_ID").apply(
        lambda g: g.loc[g["EPOCH"] >= g["EPOCH"].max() - pd.Timedelta(days=7), "ALT_KM"].median())
    hist, edges = np.histogram(last, bins=np.arange(last.min() // 5 * 5, last.max() + 10, 5))
    mode = edges[hist.argmax()] + 2.5
    return float(last[(last - mode).abs() <= 25].median())


def arrival_index(t: np.ndarray, in_band: np.ndarray, K: float) -> int | None:
    """First index that is in band and stays >= 95 % in band for the next K days of data."""
    cs = np.concatenate([[0], np.cumsum(in_band)])
    end = np.searchsorted(t, t + K, side="right")
    frac = (cs[end] - cs[np.arange(len(t))]) / np.maximum(end - np.arange(len(t)), 1)
    ok = in_band & (t + K <= t[-1]) & (frac >= 0.95)
    idx = np.flatnonzero(ok)
    return int(idx[0]) if idx.size else None


def robust_slope(t: np.ndarray, alt: np.ndarray, half: float = 3.5) -> np.ndarray:
    """km/day from the difference of medians over the following and preceding `half` days."""
    lo = np.searchsorted(t, t - half)
    hi = np.searchsorted(t, t + half, side="right")
    out = np.empty(len(t))
    for i in range(len(t)):
        before, after = alt[lo[i]:i + 1], alt[i:hi[i]]
        span = max(t[hi[i] - 1] - t[lo[i]], 1e-3)
        out[i] = (np.median(after) - np.median(before)) / (span / 2)
    return out


def classify(g: pd.DataFrame, op_alt: float, X: float, K: float, slope_thr: float,
             stranded_age: float, cutoff: pd.Timestamp) -> tuple[np.ndarray, dict]:
    t = (g["EPOCH"] - g["EPOCH"].iloc[0]).dt.total_seconds().to_numpy() / 86400
    alt = g["ALT_KM"].to_numpy()
    in_band = np.abs(alt - op_alt) <= X
    arr = arrival_index(t, in_band, K)
    slope = robust_slope(t, alt)
    state = np.where(slope >= slope_thr, "ascent", "drift").astype(object)

    status = "transit_censored"
    if arr is not None:
        state[arr:] = np.where(in_band[arr:], "operational", "excursion")
        status = "arrived"
    else:
        age = (cutoff - g["EPOCH"].iloc[0]).days
        tail = t >= t[-1] - 60
        final = np.median(alt[tail])
        spread = np.subtract(*np.percentile(alt[tail], [95, 5]))   # robust to single bad element sets
        if age >= stranded_age and spread < 10 and final < op_alt - X:
            moved = np.flatnonzero(np.abs(alt - final) > 10)
            start = moved[-1] + 1 if moved.size else 0
            state[start:] = "stranded"
            status = "stranded"

    days = lambda s: float(np.sum(np.diff(t, append=t[-1])[state == s]))
    summary = dict(status=status,
                   arrival=g["EPOCH"].iloc[arr] if arr is not None else pd.NaT,
                   days_to_arrival=t[arr] if arr is not None else np.nan,
                   days_drift=days("drift"), days_ascent=days("ascent"),
                   final_alt=float(np.median(alt[t >= t[-1] - 7])))
    return state, summary


def cadence_table(df: pd.DataFrame) -> pd.DataFrame:
    gap = df.groupby("NORAD_CAT_ID")["EPOCH"].diff().shift(-1).dt.total_seconds() / 3600
    d = pd.DataFrame({"STATE": df["STATE"], "gap_h": gap}).dropna()
    return d.groupby("STATE")["gap_h"].agg(
        n="size", median_h="median", p90_h=lambda x: x.quantile(0.9),
        frac_gt_24h=lambda x: (x > 24).mean(), frac_gt_48h=lambda x: (x > 48).mean()).round(3)


def plot_launches(df: pd.DataFrame, op_alt: float, X: float):
    for lid, g in df.groupby("LAUNCH"):
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for s, c in COLORS.items():
            m = g["STATE"] == s
            ax.scatter(g.loc[m, "EPOCH"], g.loc[m, "ALT_KM"], s=2, c=c, label=s)
        ax.axhspan(op_alt - X, op_alt + X, color="tab:blue", alpha=0.08)
        ax.set(title=f"Launch {lid}: {g['NORAD_CAT_ID'].nunique()} satellites",
               ylabel="mean altitude from mean motion (km)")
        ax.legend(markerscale=5, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / "altitude" / f"{lid}.png", dpi=120)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--X", type=float, default=15, help="band half-width, km")
    ap.add_argument("--K", type=float, default=30, help="dwell to confirm arrival, days")
    ap.add_argument("--slope", type=float, default=1.0, help="ascent threshold, km/day")
    ap.add_argument("--stranded-age", type=float, default=365, help="days since first element set")
    ap.add_argument("--tag", default="", help="write states{tag}.csv and satellite_summary{tag}.csv only "
                                              "(sensitivity runs; skips the sweep, cadence table and plots)")
    args = ap.parse_args()

    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    (OUT / "altitude").mkdir(parents=True, exist_ok=True)
    cutoff = df["EPOCH"].max()
    op_alt = operational_altitude(df)
    print(f"{df['NORAD_CAT_ID'].nunique()} satellites, {len(df)} element sets, "
          f"{df['EPOCH'].min():%Y-%m-%d} .. {cutoff:%Y-%m-%d}")
    print(f"operational altitude estimated from data: {op_alt:.1f} km")

    groups = dict(tuple(df.groupby("NORAD_CAT_ID")))
    states, rows = [], []
    for sid, g in groups.items():
        s, info = classify(g, op_alt, args.X, args.K, args.slope, args.stranded_age, cutoff)
        states.append(pd.Series(s, index=g.index))
        rows.append(dict(NORAD_CAT_ID=sid, LAUNCH=g["LAUNCH"].iloc[0],
                         first_epoch=g["EPOCH"].iloc[0], **info))
    df["STATE"] = pd.concat(states)
    sats = pd.DataFrame(rows)

    df[["NORAD_CAT_ID", "EPOCH", "LAUNCH", "ALT_KM", "STATE"]].to_csv(DATA / f"states{args.tag}.csv", index=False)
    sats.to_csv(OUT / f"satellite_summary{args.tag}.csv", index=False)
    if args.tag:
        print(f"X = {args.X:g} km, K = {args.K:g} d: " + sats["status"].value_counts().to_string().replace("\n", ", "))
        return

    # Sensitivity of arrival epochs to the two analyst choices.
    arr = {}
    for X in SWEEP_X:
        for K in SWEEP_K:
            arr[f"X{X}_K{K}"] = [classify(g, op_alt, X, K, args.slope, args.stranded_age, cutoff)[1]["days_to_arrival"]
                                 for g in groups.values()]
    sens = pd.DataFrame(arr, index=list(groups))
    sens["spread_days"] = sens.max(axis=1) - sens.min(axis=1)
    sens.to_csv(OUT / "arrival_sensitivity.csv")

    cad = cadence_table(df)
    cad.to_csv(OUT / "cadence.csv")
    plot_launches(df, op_alt, args.X)

    print("\nstatus counts:\n" + sats["status"].value_counts().to_string())
    print("\nper launch (median days to arrival, n stranded):")
    print(sats.groupby("LAUNCH").agg(n=("status", "size"),
                                     arrived=("status", lambda s: (s == "arrived").sum()),
                                     stranded=("status", lambda s: (s == "stranded").sum()),
                                     med_days_to_arrival=("days_to_arrival", "median"),
                                     med_final_alt=("final_alt", "median")).round(1).to_string())
    print("\ncadence by state (gap to next element set):\n" + cad.to_string())

    # ---- go / no-go -------------------------------------------------------------------
    old = sats[(cutoff - sats["first_epoch"]).dt.days >= 180]
    resolved = (old["status"] != "transit_censored").mean() if len(old) else np.nan
    spread = sens["spread_days"].dropna()
    early = df[(df["EPOCH"] - df.groupby("NORAD_CAT_ID")["EPOCH"].transform("min")).dt.days < 30]
    early_transit = (~early["STATE"].isin(["operational", "excursion"])).mean()
    transit_gap = df.loc[df["STATE"].isin(["drift", "ascent"])]
    transit_med = cad.loc[cad.index.isin(["drift", "ascent"]), "median_h"].max()

    print("\n==== GO / NO-GO ====")
    checks = [
        ("satellites >=180 d old with a resolved state (arrived or stranded)", resolved, resolved >= 0.8, ">= 80 %"),
        ("median spread of arrival epoch across X/K sweep (days)", spread.median(), spread.median() <= 7, "<= 7 d"),
        ("element sets in first 30 d NOT labelled operational", early_transit, early_transit >= 0.9, ">= 90 %"),
        ("worst median gap between element sets in transit states (h)", transit_med, transit_med <= 24, "<= 24 h"),
    ]
    for name, val, ok, target in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {val:.2f} (target {target})")
    print(f"\ntransit-state element sets: {len(transit_gap)}; plots in {OUT / 'altitude'}")


if __name__ == "__main__":
    main()
