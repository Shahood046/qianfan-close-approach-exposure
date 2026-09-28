"""Figure 8 (Section 6.3): observed phase-error trajectories of four satellites, and the spread of turning-point intervals.

Left: satellite-level phase error epsilon (SGP4 mean argument of latitude, nearest slot of the plane's grid, wrapped to +-5 deg) at
daily sampling for the four satellites chosen by the rules in out/fig8_picks.csv (written by 19_phase_dynamics.py). Nothing is
interpolated or fitted, no turning points are marked, and gaps are days without an operational-state element set. The vertical axis is
linear within +-1 deg (the on-grid criterion of Section 3.4) and logarithmic outside it, so that the multi-degree values of satellites
still moving between slots stay on the page. epsilon is not unwrapped: at daily sampling the change between consecutive days
reaches several degrees for such satellites (close to half the 10 deg slot spacing), so an unwrapped path would not be reliable.
Right: median interval between turning points for each of the 84 satellites of Section 6.3 (7-day smoothed epsilon, at least
0.05 deg between turns; out/phase_turning_points.csv).

Inputs: data/slot_errors_daily.pkl.gz, out/satellite_summary.csv, out/fig8_picks.csv, out/phase_turning_points.csv.
Output: out/fig8_phase_trajectories.png.   Usage:  py 27_fig8_trajectories.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, ROOT

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
C_PT, C_BAND, C_HIST = "#2a2a28", "#2a78d6", "#8d8c87"
SPARSE_LAUNCH = "2024-185"


def main():
    E = pd.read_pickle(DATA / "slot_errors_daily.pkl.gz")
    E = E[E["LAUNCH"] != SPARSE_LAUNCH].sort_values(["NORAD_CAT_ID", "t"])
    sats = pd.read_csv(OUT / "satellite_summary.csv")
    sats["arrival"] = pd.to_datetime(sats["arrival"], utc=True, format="ISO8601")
    E = E.merge(sats[["NORAD_CAT_ID", "arrival"]], on="NORAD_CAT_ID", how="left")
    picks = pd.read_csv(OUT / "fig8_picks.csv")
    RV = pd.read_csv(OUT / "phase_turning_points.csv")
    has = RV[RV["reversals"] >= 2]

    tmin, tmax = E["t"].min(), E["t"].max()
    fig = plt.figure(figsize=(11.6, 8.0), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 2, width_ratios=[3.0, 1.0], hspace=0.45, wspace=0.22)
    axes = [fig.add_subplot(gs[i, 0]) for i in range(4)]
    short = {"most turning points": "most turning points", "median turning-point amplitude": "median turning-point amplitude",
             "widest range of unwrapped eps": "widest range of ε",
             "arrived in the last 90 days of the record, largest first-day |eps|": "recent arrival, largest first-day |ε|"}
    rules = {k: short.get(v, v) for k, v in zip(picks["NORAD_CAT_ID"], picks["rule"])}
    order = list(picks["NORAD_CAT_ID"])
    for i, (ax, sid) in enumerate(zip(axes, order)):
        g = E[E["NORAD_CAT_ID"] == sid]
        ax.set_facecolor(SURFACE)
        ax.axhspan(-1, 1, color=C_BAND, alpha=0.08, lw=0)
        ax.axhline(0, color=C_BAND, lw=0.8)
        ax.plot(g["t"], g["eps_um"], ".", ms=3.2, color=C_PT, mew=0)
        ax.set_yscale("symlog", linthresh=1.0, linscale=1.6)
        ax.set_ylim(-7, 7)
        ax.set_yticks([-5, -1, 0, 1, 5]); ax.set_yticklabels(["−5", "−1", "0", "1", "5"])
        ax.minorticks_off()
        ax.set_xlim(tmin - pd.Timedelta(days=5), tmax + pd.Timedelta(days=5))
        ax.set_ylabel("\u03b5 (\u00b0)", color=INK2, fontsize=9)
        ax.set_title(f"{g['OBJECT_ID'].iloc[0]}   arrived {g['arrival'].iloc[0]:%Y-%m-%d}   (selected as: {rules.get(sid, '')})",
                     fontsize=8.8, color=INK, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c9c8c3")
        ax.tick_params(colors=INK2, labelsize=8.5)
        ax.grid(axis="x", color="#e6e5e1", lw=0.6)
        if i < 3:
            ax.tick_params(labelbottom=False)
    axes[0].text(0.995, 0.06, "shaded: |\u03b5| \u2264 1\u00b0; axis linear inside, logarithmic outside", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=7.8, color=INK2)
    axes[3].set_xlabel("Date (one point per day; gaps are days without an operational-state element set)", color=INK2, fontsize=9)

    bx = fig.add_subplot(gs[:, 1])
    bx.set_facecolor(SURFACE)
    gaps = has["gap_med"].to_numpy()
    edges = np.arange(0, 95, 5)
    bx.hist(np.clip(gaps, 0, 92.5), bins=edges, color=C_HIST, edgecolor=SURFACE, lw=0.8)
    med = np.median(gaps)
    p10, p90 = np.percentile(gaps, [10, 90])
    bx.axvline(med, color=INK, lw=1.0, ls=(0, (3, 2)))
    bx.text(med + 1.5, bx.get_ylim()[1] * 0.97, f"median {med:.0f} d", fontsize=8.5, color=INK, va="top")
    bx.set_xlabel("Median interval between turning points (d), per satellite", color=INK2, fontsize=9)
    bx.set_ylabel("Satellites", color=INK2, fontsize=9)
    bx.set_title(f"(b) Turning-point intervals, {len(gaps)} satellites", fontsize=8.8, color=INK, loc="left")
    for s in ("top", "right"):
        bx.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        bx.spines[s].set_color("#c9c8c3")
    bx.tick_params(colors=INK2, labelsize=8.5)
    bx.grid(axis="y", color="#e6e5e1", lw=0.6)
    bx.set_axisbelow(True)
    fig.savefig(OUT / "fig8_phase_trajectories.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")

    lines = ["# Figure 8 numbers (27_fig8_trajectories.py)", "",
             f"Satellites drawn: {', '.join(E[E['NORAD_CAT_ID'].isin(order)].drop_duplicates('NORAD_CAT_ID')['OBJECT_ID'])} "
             f"(rules in out/fig8_picks.csv).",
             f"Panel (b): {len(gaps)} satellites with at least two turning points (of {len(RV)} with the required record); "
             f"median interval {med:.1f} d, P10-P90 {p10:.0f}-{p90:.0f} d (bins of 5 d; the last bin includes values above 90 d)."]
    (ROOT / "FIG8_TRAJECTORIES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
