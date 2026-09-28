"""Figure 9 (Section 7): where the population lies, and where the events occur.

(a) Primary pair-windows by reconstructed nominal-margin bin (Table 10): the number of pair-windows per bin and the observed event
    rate per 1,000 pair-windows with Poisson 95% intervals. The bins are the discrete Table 10 classes (unequal widths in km), drawn
    as separate bars and points, not as a continuous curve.
(b) Threshold-free concentration: cumulative share of observed sub-10 km events within the smallest-margin fraction of pair-windows,
    pooled, before and from April 2026, with 7-day moving-block bootstrap bands.

Inputs: data/pair_windows_daily.pkl.gz (SGP4 mean phase). Uses table() from 22_margin_table_fig.py and share_at/block_boot from
14_locked_claims.py, so the numbers are those of MARGIN_TABLE.md and LOCKED_CLAIMS.md.  Output: out/fig9_concentration.png.

Usage:  py 26_fig9_panels.py
"""
from __future__ import annotations

import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, ROOT


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


tm = load("tm", "22_margin_table_fig.py")
lc = tm.lc
C_POOLED, C_EARLY, C_RECENT = tm.C_POOLED, tm.C_EARLY, tm.C_RECENT
INK, INK2, SURFACE = tm.INK, tm.INK2, tm.SURFACE
GRID = "#e6e5e1"


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main():
    P = pd.read_pickle(DATA / "pair_windows_daily.pkl.gz")
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    G = P[P["good"] & P["std_angle"]].copy()
    T, _ = tm.table(G)
    labels = ["0-5", "5-10", "10-15", "15-20", "20-30", "30-40", "40-60", "60-100", "100-200", ">=200"]
    labels = [l.replace("-", "\u2013").replace(">=", "\u2265") for l in labels]
    x = np.arange(len(T))
    overall = 1000 * G["obs10"].sum() / len(G)

    fig = plt.figure(figsize=(12.2, 5.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1], height_ratios=[1, 1.15], hspace=0.12, wspace=0.34)
    a1 = fig.add_subplot(gs[0, 0]); a2 = fig.add_subplot(gs[1, 0], sharex=a1); b = fig.add_subplot(gs[:, 1])

    # (a1) population
    style(a1)
    a1.bar(x, T["pair_windows"], width=0.7, color="#b8b7b2", zorder=2)
    a1.set_yscale("log"); a1.set_ylim(100, 3e7)
    a1.set_ylabel("Pair-windows", color=INK2, fontsize=9)
    for xi, n in zip(x, T["pair_windows"]):
        a1.text(xi, n * 1.25, f"{n:,}", ha="center", va="bottom", fontsize=7.2, color=INK2)
    plt.setp(a1.get_xticklabels(), visible=False)
    a1.set_title("(a) Population and event rate by nominal-margin bin", fontsize=10, color=INK, loc="left")

    # (a2) observed event rate, points with Poisson intervals
    style(a2)
    r = T["rate"].to_numpy()
    a2.errorbar(x, r, yerr=[r - T["lo"], T["hi"] - r], fmt="o", color=C_POOLED, ecolor=C_POOLED, ms=5.5, elinewidth=1.3, capsize=2.5, zorder=3)
    a2.axhline(overall, color=INK2, lw=0.9, ls=(0, (2, 2)), zorder=1)
    a2.text(len(T) - 0.55, overall * 1.25, f"all primary pair-windows: {overall:.2f}", ha="right", va="bottom", fontsize=8, color=INK2)
    a2.set_yscale("log"); a2.set_ylim(0.004, 1000)
    a2.set_ylabel("Events per 1,000 pair-windows", color=INK2, fontsize=9)
    a2.set_xticks(x); a2.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right"); a2.minorticks_off(); a1.minorticks_off()
    a2.set_xlabel("Nominal margin at the window centre (km); Table 10 bins, unequal widths", color=INK2, fontsize=8.5)
    a2.set_xlim(-0.6, len(T) - 0.4)

    # (b) concentration
    style(b)
    fr = np.unique(np.r_[np.geomspace(0.0005, 0.2, 60), 0.005, 0.01, 0.015])
    early = G["t0"] < lc.EARLY_END
    sets = [("Pooled", G, C_POOLED, "-"), ("Before April 2026", G[early], C_EARLY, "--"), ("From April 2026", G[~early], C_RECENT, ":")]
    for name, X, col, ls in sets:
        pt = lc.share_at(X["nominal"].to_numpy(), X["obs10"].to_numpy(), fr)
        sh, _ = lc.block_boot(X, fr, seed=2)
        lo, hi = np.percentile(sh, [2.5, 97.5], axis=0)
        b.fill_between(100 * fr, 100 * lo, 100 * hi, color=col, alpha=0.14, lw=0)
        b.plot(100 * fr, 100 * pt, color=col, lw=2, ls=ls, label=f"{name} ({int(X['obs10'].sum())} events)")
    cut = np.quantile(G["nominal"], 0.01)
    p1 = lc.share_at(G["nominal"].to_numpy(), G["obs10"].to_numpy(), [0.01])[0]
    b.axvline(1, color=INK2, lw=0.8, ls=(0, (2, 2)))
    b.annotate(f"1% of pair-windows\n(nominal margin < {cut:.0f} km):\n{100 * p1:.1f}% of events", (1, 84), xytext=(6, 0),
               textcoords="offset points", fontsize=8.5, color=INK, va="center")
    leg = b.legend(loc="lower right", frameon=False, fontsize=8.5, handlelength=3)
    for t in leg.get_texts():
        t.set_color(INK2)
    b.set_xscale("log"); b.set_xlim(100 * fr.min(), 100 * fr.max()); b.set_ylim(0, 101)
    b.set_xlabel("Smallest-margin pair-windows included (% of primary pair-windows)", color=INK2, fontsize=8.5)
    b.set_ylabel("Events captured (%)", color=INK2, fontsize=9)
    b.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    b.set_title("(b) Concentration of sub-10 km events", fontsize=10, color=INK, loc="left")

    fig.savefig(OUT / "fig9_concentration.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    lines = ["# Figure 9 numbers (26_fig9_panels.py)", "",
             f"Primary pair-windows {len(G):,}; events {int(G['obs10'].sum())}; overall rate {overall:.3f} per 1,000.", "",
             "Panel (a) values are Table 10 (MARGIN_TABLE.md). Panel (b): "
             f"1% cut at nominal margin {cut:.1f} km, pooled share {100 * p1:.1f}% of events; "
             f"before April 2026 {int(G[early]['obs10'].sum())} events, from April 2026 {int(G[~early]['obs10'].sum())} events."]
    (ROOT / "FIG9_PANELS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
