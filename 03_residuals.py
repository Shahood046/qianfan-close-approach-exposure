"""Empirical TLE prediction residuals, by orbital-transition state and propagation age.

For each element set i of a satellite, propagate it with SGP4 to the epoch of every later element
set j (up to 7 days ahead) and difference against element set j evaluated at its own epoch.
The residual is expressed in the radial / in-track / cross-track frame of j.

This is a *prediction-residual* model, not a true error model: element set j has its own fit error,
and in transit states the residual also contains unmodelled thrust. That is exactly why it is
computed separately per state.

Usage:  py 03_residuals.py [--max-sats 60]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from common import DATA, OUT, load_gp, load_states, rtn, satrec, to_jd

AGE_BINS_H = [0, 6, 12, 24, 48, 96, 168]
MAX_AGE_H = AGE_BINS_H[-1]


def residuals_for(g: pd.DataFrame) -> pd.DataFrame:
    sats = [satrec(row) for _, row in g.iterrows()]
    jd, fr = to_jd(g["EPOCH"])
    # "Truth" proxy: each element set evaluated at its own epoch.
    truth = [s.sgp4(a, b) for s, a, b in zip(sats, jd, fr)]
    ok = np.array([e == 0 for e, _, _ in truth])
    r_t = np.array([r for _, r, _ in truth])
    v_t = np.array([v for _, _, v in truth])
    t_h = (g["EPOCH"] - g["EPOCH"].iloc[0]).dt.total_seconds().to_numpy() / 3600

    out = []
    for i, s in enumerate(sats):
        j = np.arange(i + 1, np.searchsorted(t_h, t_h[i] + MAX_AGE_H, side="right"))
        j = j[ok[j]]
        if not ok[i] or j.size == 0:
            continue
        e, r, _ = s.sgp4_array(jd[j], fr[j])
        good = e == 0
        j, r = j[good], r[good]
        d = rtn(r - r_t[j], r_t[j], v_t[j])
        out.append(pd.DataFrame({"NORAD_CAT_ID": g["NORAD_CAT_ID"].iloc[0], "EPOCH": g["EPOCH"].iloc[i],
                                 "age_h": t_h[j] - t_h[i], "dR": d[:, 0], "dT": d[:, 1], "dN": d[:, 2]}))
    return pd.concat(out) if out else pd.DataFrame()


def pick_sample(df: pd.DataFrame, states: pd.DataFrame, n: int) -> list[int]:
    """Spread the sample over launches, making sure satellites that spent time in transit are included."""
    transit = states[states["STATE"].isin(["drift", "ascent"])].groupby("NORAD_CAT_ID").size()
    per_launch = max(1, n // df["LAUNCH"].nunique())
    ids = []
    for _, g in df.groupby("LAUNCH"):
        cand = transit.reindex(g["NORAD_CAT_ID"].unique()).fillna(0).sort_values(ascending=False)
        ids += list(cand.index[:per_launch])
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-sats", type=int, default=60)
    args = ap.parse_args()

    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    if states is None:
        raise SystemExit("Run 02_states.py first.")
    ids = pick_sample(df, states, args.max_sats)
    print(f"computing residuals for {len(ids)} satellites")

    res = pd.concat([residuals_for(df[df["NORAD_CAT_ID"] == sid]) for sid in ids])
    res = res.merge(states[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"], how="left")
    res["dr"] = np.sqrt(res["dR"]**2 + res["dT"]**2 + res["dN"]**2)
    res["age_bin"] = pd.cut(res["age_h"], AGE_BINS_H)
    res.to_csv(OUT / "residual_pairs.csv.gz", index=False)

    q = lambda p: (lambda x: x.quantile(p))
    summ = res.groupby(["STATE", "age_bin"], observed=True).agg(
        n=("dr", "size"),
        bias_R=("dR", "mean"), bias_T=("dT", "mean"), bias_N=("dN", "mean"),
        sd_R=("dR", "std"), sd_T=("dT", "std"), sd_N=("dN", "std"),
        dr_p50=("dr", q(0.5)), dr_p95=("dr", q(0.95)),
        kurt_T=("dT", lambda x: kurtosis(x, nan_policy="omit")),
    ).round(3)
    summ.to_csv(OUT / "residual_summary.csv")
    print(summ.to_string())

    fig, ax = plt.subplots(figsize=(7, 5))
    mids = res.groupby(["STATE", "age_bin"], observed=True).agg(age=("age_h", "median"),
                                                               p50=("dr", "median"), p95=("dr", q(0.95)))
    for s, g in mids.groupby(level=0):
        line, = ax.plot(g["age"], g["p50"], "o-", label=f"{s} median")
        ax.plot(g["age"], g["p95"], "o--", color=line.get_color(), alpha=0.6, label=f"{s} 95th pct")
    for thr in (5, 10, 20):
        ax.axhline(thr, color="grey", lw=0.8, ls=":")
    ax.set(xscale="log", yscale="log", xlabel="propagation age (h)", ylabel="|position residual| (km)",
           title="SGP4 prediction residual vs next element set")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "residuals_vs_age.png", dpi=130)

    print("\n==== GO / NO-GO ====")
    for s in ["operational", "drift", "ascent"]:
        row = summ.loc[(s, pd.Interval(12, 24, closed="right"))] if (s, pd.Interval(12, 24, closed="right")) in summ.index else None
        if row is None:
            print(f"[----] no {s} pairs in the 12-24 h bin")
            continue
        ok = row["dr_p95"] < 5
        print(f"[{'PASS' if ok else 'WARN'}] {s}: 95th-pct residual at 12-24 h = {row['dr_p95']:.2f} km "
              f"(needs to be well below the 5 km threshold for 5 km counts to mean anything)")
    print(f"figure: {OUT / 'residuals_vs_age.png'}")


if __name__ == "__main__":
    main()
