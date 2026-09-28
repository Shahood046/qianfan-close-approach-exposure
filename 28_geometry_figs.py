"""Figures 4, 5 and 6 (Section 4), drawn from the tables saved by 17_geometry.py (no recomputation of the analysis).

Fig. 4  reconstructed star geometry: (a) plane orientations from the measured RAAN (polar view, planes labelled P1-P9 in order of
        RAAN as in Table 5), (b) the crossing geometry behind the separation model of Section 5 (schematic, not to scale),
        (c) what a grid offset of about 0 deg (even plane step) and about half a slot (odd step) means at a crossing (schematic;
        offsets are the medians quoted in Section 4.5).
Fig. 5  (a) pooled in-plane coherence R(s) at the end of the record (10 deg grid: R = 0.970), (b) R(10 deg) at the 30-day snapshots.
Fig. 6  grid-to-grid offset at the crossing for every standard-angle plane pair of populated planes at each snapshot, even and odd steps.

Inputs: out/final_planes.csv, out/coherence_scan.csv, out/geometry_snapshots.csv,
        out/geometry_snapshot_offsets.csv.   Outputs: out/fig4_geometry.png, out/fig5_slot_grid.png, out/fig6_grid_offset.png,
        FIG456_NUMBERS.md.      Usage:  py 28_geometry_figs.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc

from common import OUT, ROOT

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
BLUE, ORANGE, GREY, GRID = "#2a78d6", "#eb6834", "#a9a8a3", "#e6e5e1"


def style(ax, grid=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def fig4(T):
    fig = plt.figure(figsize=(12.0, 4.9), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.0], wspace=0.22)
    # ---- (a) polar view of plane orientations
    ax = fig.add_subplot(gs[0]); ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(SURFACE)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ec="#c9c8c3", lw=1))
    ax.plot(0, 0, "o", color=INK2, ms=3)
    ax.text(0.05, -0.12, "pole", fontsize=8, color=INK2, bbox=dict(fc=SURFACE, ec="none", pad=1.0), zorder=5)
    for r in T.itertuples():
        th = np.radians(r.raan)
        col, ls = (GREY, (0, (3, 2))) if r.sparse else (BLUE, "-")
        ax.plot([0, np.cos(th)], [0, np.sin(th)], color=col, lw=1.4 if r.sparse else 2.0, ls=ls, zorder=2)
        ax.plot([0, -np.cos(th)], [0, -np.sin(th)], color=col, lw=1.0, ls=ls, alpha=0.35, zorder=1)
        ax.plot(np.cos(th), np.sin(th), "o", color=col, ms=5, zorder=3)
        ax.text(1.14 * np.cos(th), 1.14 * np.sin(th), f"{r.label}\n({r.n})", ha="center", va="center", fontsize=8.2,
                color=INK2 if r.sparse else INK)
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.35, 1.4)
    ax.set_title("(a) Plane orientations (polar view)", fontsize=10, color=INK, loc="left")
    ax.text(0, -1.33, "Bold spoke and dot: ascending-node direction (measured RAAN);\nfaint spoke: opposite (descending) node. Labels: plane (satellites).\nDashed: sparse plane. Not a design drawing.",
            ha="center", va="top", fontsize=7.8, color=INK2)
    # ---- (b) crossing geometry
    bx = fig.add_subplot(gs[1]); bx.set_aspect("equal"); bx.axis("off"); bx.set_facecolor(SURFACE)
    g = np.radians(20.5 / 2 * 1.9)            # drawn half-angle (exaggerated for legibility)
    d1 = np.array([np.cos(g), np.sin(g)]); d2 = np.array([np.cos(g), -np.sin(g)])
    for d, c in ((d1, BLUE), (d2, ORANGE)):
        bx.plot([-d[0], d[0]], [-d[1], d[1]], color=c, lw=1.8)
    a_pos, b_pos = -0.62 * d1, 0.48 * d2
    bx.plot(*a_pos, "o", color=BLUE, ms=8); bx.plot(*b_pos, "o", color=ORANGE, ms=8)
    bx.annotate("", xy=b_pos, xytext=a_pos, arrowprops=dict(arrowstyle="<->", color=INK, lw=1.0, shrinkA=6, shrinkB=6))
    bx.plot(0, 0, "x", color=INK, ms=6)
    bx.text(0.0, 0.10, "crossing point", fontsize=8, color=INK2, ha="center", va="bottom")
    bx.add_patch(Arc((0, 0), 0.9, 0.9, theta1=-np.degrees(g), theta2=np.degrees(g), color=INK2, lw=1))
    bx.text(0.50, 0.0, "\u03b3", fontsize=11, color=INK2, va="center")
    bx.text(0, -0.72, "Two satellites reach the crossing of their planes with an\narc offset \u0394\u03c6 and a radial offset \u0394r:\n"
            "d \u2248 \u221a[(r \u0394\u03c6 cos(\u03b3/2))\u00b2 + \u0394r\u00b2]   (Section 5.1)\n\nSchematic, not to scale (\u03b3 exaggerated).",
            ha="center", va="top", fontsize=8.2, color=INK)
    bx.set_xlim(-1.2, 1.2); bx.set_ylim(-1.55, 0.8)
    bx.set_title("(b) Crossing geometry", fontsize=10, color=INK, loc="left")
    # ---- (c) slot grids at a crossing
    cx = fig.add_subplot(gs[2]); cx.set_facecolor(SURFACE); cx.axis("off")
    rows = [("Even plane step (k = 2, 4, ...)", 0.24, 1.95), ("Odd plane step (k = 1, 3, ...)", 4.67, 0.55)]
    for title, off, y in rows:
        cx.text(-4, y + 0.62, title, fontsize=8.8, color=INK, ha="left")
        cx.plot([-5, 55], [y + 0.25, y + 0.25], color="#c9c8c3", lw=0.8)
        cx.plot([-5, 55], [y - 0.25, y - 0.25], color="#c9c8c3", lw=0.8)
        for k in range(6):
            cx.plot(10 * k, y + 0.25, "o", color=BLUE, ms=8)
            cx.plot(10 * k + off, y - 0.25, "D", color=ORANGE, ms=7)
        cx.text(-5.5, y + 0.25, "plane A", ha="right", va="center", fontsize=8, color=INK2)
        cx.text(-5.5, y - 0.25, "plane B", ha="right", va="center", fontsize=8, color=INK2)
        cx.text(55, y - 0.62, f"grid offset at the crossing: median {off:.2f}\u00b0", ha="right", fontsize=8, color=INK2)
    cx.set_xlim(-16, 58); cx.set_ylim(-0.9, 2.8)
    cx.set_title("(c) Slot grids at a crossing (10\u00b0 spacing)", fontsize=10, color=INK, loc="left")
    cx.text(21, -0.55,"Offsets are the end-of-record medians of Section 4.5 (Table 6).\nWhether occupied slots coincide or interleave also\ndepends on slot occupancy (Section 4.5).",
            ha="center", va="top", fontsize=7.8, color=INK2)
    fig.savefig(OUT / "fig4_geometry.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def fig5(C, S):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.22), facecolor=SURFACE)
    style(a, "y")
    a.plot(C["s_deg"], C["R"], color=BLUE, lw=1.8)
    a.axhline(0.9, color=INK2, lw=0.8, ls=(0, (2, 2)))
    for s, xy in ((5, (5.7, 0.74)), (10, (10.9, 0.95)), (20, (20.7, 0.66))):
        r = float(C.loc[np.isclose(C["s_deg"], s), "R"].iloc[0])
        a.plot(s, r, "o", color=INK, ms=5)
        a.annotate(f"s = {s}\u00b0: R = {r:.3f}", (s, r), xytext=xy, fontsize=8.5, color=INK, ha="left",
                   arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7, shrinkA=0, shrinkB=4))
    a.text(24.8, 0.915, "R = 0.9", ha="right", fontsize=8, color=INK2)
    a.set_xlim(4, 25); a.set_ylim(0, 1.08)
    a.set_xlabel("Trial slot spacing s (\u00b0)", color=INK2, fontsize=9)
    a.set_ylabel("Pooled coherence R(s)", color=INK2, fontsize=9)
    a.set_title("(a) Spacing scan, end of record", fontsize=10, color=INK, loc="left")
    style(b, "y")
    d = pd.to_datetime(S["date"])
    b.plot(d, S["R10"], "o-", color=BLUE, ms=4.5, lw=1.2)
    b.axhline(0.9, color=INK2, lw=0.8, ls=(0, (2, 2)))
    b.annotate(f"{S['planes'].iloc[0]} planes populated", (d.iloc[0], S["R10"].iloc[0]), xytext=(8, -3), textcoords="offset points",
               fontsize=7.8, color=INK2, ha="left")
    b.annotate(f"{S['planes'].iloc[-1]} planes populated", (d.iloc[-1], S["R10"].iloc[-1]), xytext=(-6, 10), textcoords="offset points",
               fontsize=7.8, color=INK2, ha="right", bbox=dict(fc=SURFACE, ec="none", pad=1.0))
    b.set_ylim(0.6, 1.02)
    b.set_ylabel("R(10\u00b0)", color=INK2, fontsize=9)
    b.set_xlabel("Date (30-day snapshots)", color=INK2, fontsize=9)
    b.set_title("(b) Coherence at 10\u00b0 over time", fontsize=10, color=INK, loc="left")
    plt.setp(b.get_xticklabels(), rotation=30, ha="right")
    fig.savefig(OUT / "fig5_slot_grid.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def fig6(O):
    O = O[O["std"] & ~O["sparse"]].copy()
    O["date"] = pd.to_datetime(O["date"]); O["abs"] = O["offset_deg"].abs()
    ev, od = O[O["k"] % 2 == 0], O[O["k"] % 2 == 1]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=SURFACE)
    style(ax, "y")
    ax.axhline(5, color=INK2, lw=0.8, ls=(0, (2, 2)))
    ax.text(ev["date"].min(), 5.12, "half a slot (5\u00b0)", fontsize=8, color=INK2, va="bottom")
    ax.scatter(ev["date"], ev["abs"], s=22, marker="o", color=BLUE, alpha=0.75, lw=0, label=f"Even plane step (n = {len(ev)})")
    ax.scatter(od["date"], od["abs"], s=26, marker="D", color=ORANGE, alpha=0.75, lw=0, label=f"Odd plane step (n = {len(od)})")
    ax.set_ylim(-0.2, 5.6)
    ax.set_ylabel("|grid offset at the crossing| (\u00b0)", color=INK2, fontsize=9)
    ax.set_xlabel("Date (30-day snapshots; each marker is one plane pair)", color=INK2, fontsize=9)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.62), frameon=False, fontsize=8.8)
    ax.set_title("Grid-to-grid offset at the crossing, standard-angle pairs of populated planes", fontsize=10, color=INK, loc="left")
    fig.savefig(OUT / "fig6_grid_offset.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    return ev, od


def main():
    T = pd.read_csv(OUT / "final_planes.csv")
    C = pd.read_csv(OUT / "coherence_scan.csv")
    S = pd.read_csv(OUT / "geometry_snapshots.csv")
    O = pd.read_csv(OUT / "geometry_snapshot_offsets.csv")
    fig4(T); fig5(C, S)
    ev, od = fig6(O)
    later = S[pd.to_datetime(S["date"]) >= "2025-05-31"]
    r = lambda s: float(C.loc[np.isclose(C["s_deg"], s), "R"].iloc[0])
    lines = ["# Figures 4-6 numbers (28_geometry_figs.py; tables from 17_geometry.py)", "",
             f"- Fig. 4a: {len(T)} plane positions ({int((~T['sparse']).sum())} populated, {int(T['sparse'].sum())} sparse), n per plane "
             + ", ".join(f"{q.label} {q.n}" for q in T.itertuples()),
             f"- Fig. 5a: R(10) = {r(10):.3f}; R(5) = {r(5):.3f}; R(20) = {r(20):.3f}",
             f"- Fig. 5b: R(10) from 2025-05-31 onward ranges {later['R10'].min():.3f}-{later['R10'].max():.3f}",
             f"- Fig. 6: even-step |offset| median {ev['abs'].median():.2f} deg (P90 {ev['abs'].quantile(.9):.2f}), odd-step median {od['abs'].median():.2f}, "
             f"minimum {od['abs'].min():.2f}; {len(ev) + len(od)} plane-pair snapshots (even {len(ev)}, odd {len(od)}), {O['date'].nunique()} snapshots"]
    (ROOT / "FIG456_NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
