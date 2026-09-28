"""Figure 7: analytical crossing model against SGP4, V4 versus V6 (Section 5.2, Table 7).

Runs part A of 09_model_checks.py (which writes out/model_vs_sgp4_pairs.csv for the 113 inter-plane pairs with SGP4 minimum
<= 50 km at the reference epoch) unless --plot-only is given, then draws two panels on shared axes, V4 (phase in the osculating
argument of latitude, eccentricity vectors) and V6 (phase in SGP4's mean argument of latitude). Pairs involving P4 or P6 are marked.

Usage:  py 25_fig7_model.py [--plot-only]
"""
from __future__ import annotations

import argparse
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, ROOT, load_gp, load_states

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
C_MAIN, C_FLAG = "#2a78d6", "#eb6834"


def load_09():
    spec = importlib.util.spec_from_file_location("mc", ROOT / "09_model_checks.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    return mc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        mc = load_09()
        df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
        mc.part_a(df, load_states(), df["EPOCH"].max().floor("h") - pd.Timedelta(days=1))
    P = pd.read_csv(OUT / "model_vs_sgp4_pairs.csv")
    flag = P["plane_1"].isin(["P4", "P6"]) | P["plane_2"].isin(["P4", "P6"])
    stats = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.5), sharex=True, sharey=True, facecolor=SURFACE)
    lo, hi = 0.25, 80
    for ax, (col, title) in zip(axes, (("V4", "V4: osculating phase"), ("V6", "V6: mean phase (used in this paper)"))):
        d, t = P[col].to_numpy(), P["sgp4_dmin_km"].to_numpy()
        r = np.corrcoef(np.log(d + 0.1), np.log(t + 0.1))[0, 1]
        err = np.abs(d - t)
        stats[col] = (r, np.median(err), np.percentile(err, 90), err.max())
        ax.set_facecolor(SURFACE)
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.8, ls="--", zorder=1)
        ax.scatter(t[~flag], d[~flag], s=16, color=C_MAIN, alpha=0.8, lw=0, zorder=2, label="other plane pairs")
        ax.scatter(t[flag], d[flag], s=26, marker="D", facecolor="none", edgecolor=C_FLAG, lw=1.2, zorder=3,
                   label="pairs involving P4 or P6")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("SGP4 minimum distance over one orbit (km)", color=INK2)
        ax.text(0.97, 0.05, f"r = {r:.3f}\nmedian |error| {np.median(err):.2f} km\n90th percentile {np.percentile(err, 90):.1f} km",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(color="#e5e5e2", lw=0.6)
        ax.tick_params(colors=INK2)
    axes[0].set_ylabel("Analytical crossing model (km)", color=INK2)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_model_vs_sgp4.png", dpi=200, facecolor=SURFACE)
    lines = ["# Figure 7 numbers (25_fig7_model.py)", "",
             f"{len(P)} inter-plane pairs with SGP4 minimum <= 50 km; reference epoch from 09_model_checks.py part A.", "",
             "| version | r (log d) | median abs error (km) | 90th pct (km) | max (km) |", "|---|---|---|---|---|"]
    lines += [f"| {k} | {v[0]:.3f} | {v[1]:.2f} | {v[2]:.2f} | {v[3]:.1f} |" for k, v in stats.items()]
    big = P.assign(err4=(P["V4"] - P["sgp4_dmin_km"]).abs()).nlargest(6, "err4")
    lines += ["", "Six largest V4 errors:", "", "| pair | planes | SGP4 (km) | V4 (km) | V6 (km) |", "|---|---|---|---|---|"]
    lines += [f"| {r.id_1}/{r.id_2} | {r.plane_1}-{r.plane_2} | {r.sgp4_dmin_km:.1f} | {r.V4:.1f} | {r.V6:.1f} |" for r in big.itertuples()]
    (ROOT / "FIG7_MODEL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
