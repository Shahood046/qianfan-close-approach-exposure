"""Appendix figure (working number Fig. A1): the phase variable and the crossing projection used by the separation model.

Purely methodological: it draws the formulas of Sections 5.1 and 5.4 and contains no empirical result. The only measured inputs are the
plane eccentricities of Table 5 / Section 4.4 (e about 0.0002, 0.0008-0.0009 and 0.0013) used to size the equation of centre, and the
mean orbital radius of about 7,447 km (Section 5.4). Panel (b) is exact geometry for two straight paths of equal speed that meet at
angle gamma (drawn for gamma = 41 deg, a plane step k = 2 crossing), not a simulation.

(a) osculating minus mean argument of latitude, 2 e sin(u - w), for three eccentricities;
(b) two satellites at the moment of closest approach: each is v*tau/2 from the crossing point and the separation is v*tau*cos(gamma/2);
(c) the chain osculating u -> mean u-bar -> phase offset at the crossing -> separation, and where the slot error enters.

Output: out/figA1_appendix_projection.png.     Usage:  py 29_appendix_figA1.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyBboxPatch

from common import OUT

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
R_KM = 7447.0


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def flow(ax):
    ax.axis("off"); ax.set_xlim(0, 116); ax.set_ylim(-8, 30)
    boxes = ["Osculating u\n(from the SGP4\nstate)",
             "Remove the\nequation of centre\nū = u − 2e sin(u − ω)",
             "Mean argument\nof latitude ū\n(advances nearly\nuniformly in time)",
             "Phase offset at\nthe crossing\nΔφ = [ū₁ − ū(u*₁)]\n− [ū₂ − ū(u*₂)]",
             "Crossing separation\nd ≈ √[(r̄ Δφ cos γ/2)²\n+ Δr²]"]
    xc = [10, 34, 58, 82, 106]
    for txt, x in zip(boxes, xc):
        ax.add_patch(FancyBboxPatch((x - 10, 4), 20, 22, boxstyle="round,pad=0.1,rounding_size=1.2", fc="#f1f0ec", ec="#c9c8c3", lw=1))
        ax.text(x, 15, txt, ha="center", va="center", fontsize=7.6, color=INK)
    for x in xc[:-1]:
        ax.annotate("", xy=(x + 13.6, 15), xytext=(x + 10.4, 15), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1))
    ax.text(0, -3.5, "Slot error ε = ū minus the nearest slot of the plane's 10° grid (measured in SGP4's mean argument of latitude, Sections 4.2, 5.4). "
            "The nominal margin is the same expression\nwith each satellite's phase moved onto its slot (Section 4.6). The mean phase is used because a difference in ū maps linearly onto a "
            "difference in arrival time at the crossing (Section 5.4).", fontsize=7.6, color=INK2, va="center")
    ax.set_title("(c) Chain used throughout the paper", fontsize=10, color=INK, loc="left")


def main():
    fig = plt.figure(figsize=(11.6, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.62, 1.0], hspace=0.28, wspace=0.18)
    flow(fig.add_subplot(gs[0, :]))

    # (a) equation of centre
    a = fig.add_subplot(gs[1, 0]); style(a)
    x = np.linspace(0, 360, 721)
    for e, col, ls, lab in ((0.0013, ORANGE, "-", "e ≈ 0.0013 (P4, P6)"), (0.0009, BLUE, "--", "e ≈ 0.0009 (P1, P5)"),
                            (0.0002, GREEN, ":", "e ≈ 0.0002 (P2, P7, P8, P9)")):
        a.plot(x, np.degrees(2 * e * np.sin(np.radians(x))), color=col, lw=2.0, ls=ls, label=lab)
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xlim(0, 360); a.set_xticks(range(0, 361, 90))
    a.set_ylim(-0.19, 0.19)
    a.set_xlabel("u − ω (°)", color=INK2, fontsize=9)
    a.set_ylabel("u − ū = 2e sin(u − ω) (°)", color=INK2, fontsize=9)
    a.annotate("largest amplitude 2e = 0.15°,\nabout 19 km along-track at r ≈ 7,447 km", (90, np.degrees(2 * 0.0013)), xytext=(112, 0.135),
               fontsize=8.2, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    a.legend(loc="lower left", frameon=False, fontsize=8.2)
    a.set_title("(a) Osculating minus mean argument of latitude", fontsize=10, color=INK, loc="left")

    # (b) closest approach of two equal-speed straight paths
    b = fig.add_subplot(gs[1, 1]); b.set_aspect("equal"); b.axis("off"); b.set_facecolor(SURFACE)
    g = np.radians(41.0 / 2)
    d1 = np.array([np.cos(g), np.sin(g)]); d2 = np.array([np.cos(g), -np.sin(g)])
    L = 1.15
    for d, c in ((d1, BLUE), (d2, ORANGE)):
        b.plot([-L * d[0], L * d[0]], [-L * d[1], L * d[1]], color=c, lw=1.8, zorder=1)
        b.annotate("", xy=0.98 * L * d, xytext=0.70 * L * d, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6), zorder=2)
    delta = 0.55
    p1, p2 = delta * d1, -delta * d2
    b.plot(0, 0, "x", color=INK, ms=7, zorder=4)
    b.text(-0.06, -0.10, "crossing point", fontsize=8, color=INK2, va="top", ha="right")
    b.plot(*p1, "o", color=BLUE, ms=9, zorder=5); b.plot(*p2, "o", color=ORANGE, ms=9, zorder=5)
    b.annotate("", xy=p1, xytext=p2, arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2, shrinkA=7, shrinkB=7), zorder=3)
    b.text(0, p1[1] + 0.08, "d = vτ cos(γ/2)", ha="center", va="bottom", fontsize=9.5, color=INK)
    for p, c, lab in ((p1, BLUE, "satellite 1\n(passed the crossing)"), (p2, ORANGE, "satellite 2\n(still to reach it)")):
        pass
    b.text(p1[0] + 0.07, p1[1] - 0.05, "satellite 1: vτ/2\nbeyond the crossing", fontsize=8, color=BLUE, va="top")
    b.text(p2[0] - 0.07, p2[1] - 0.05, "satellite 2: vτ/2\nbefore the crossing", fontsize=8, color=ORANGE, va="top", ha="right")
    b.add_patch(Arc((0, 0), 0.75, 0.75, theta1=-np.degrees(g), theta2=np.degrees(g), color=INK2, lw=1))
    b.text(0.42, 0.0, "γ", fontsize=11, color=INK2, va="center")
    b.text(0, -1.02, "Satellite 2 reaches the crossing a time τ after satellite 1 (τ ≈ Δφ/n, so vτ ≈ rΔφ).\n"
           "For equal speeds the closest approach is at the midpoint in time.\nThe radial offset Δr is perpendicular to both paths\nand adds in quadrature (Section 5.1).\n"
           "Drawn for γ = 41° (plane step k = 2); not to scale.", ha="center", va="top", fontsize=8.2, color=INK)
    b.set_xlim(-1.3, 1.3); b.set_ylim(-1.75, 0.85)
    b.set_title("(b) Closest approach at a plane crossing", fontsize=10, color=INK, loc="left")
    fig.savefig(OUT / "figA1_appendix_projection.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print("written", OUT / "figA1_appendix_projection.png")


if __name__ == "__main__":
    main()
