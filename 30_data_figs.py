"""Figures 1, 2 and 3 (Section 3), drawn from the saved state table and residual summary (no recomputation of the analysis).

Fig. 1  build-out: satellites per orbital-transition state over time (from data/states.csv), with the first element set of each launch marked.
Fig. 2  altitude histories by state for four launches (from data/states.csv): the first launch, the launch with most satellites stranded
        (2024-185), and two later launches. Time is days since the launch's first element set.
Fig. 3  SGP4 prediction residual against propagation age by state (from out/residual_summary.csv; medians and 95th percentiles per age
        bin). The stranded state, represented by one satellite, is omitted (Table 3 keeps its row).

Inputs: data/states.csv, out/residual_summary.csv.  Outputs: out/fig1_buildout.png, out/fig2_altitude_states.png,
out/fig3_residuals.png, FIG123_NUMBERS.md.     Usage:  py 30_data_figs.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from common import DATA, OUT, ROOT, load_states

INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e1"
STATE_ORDER = ["operational", "excursion", "drift", "ascent", "stranded"]
# Okabe-Ito based, fixed per state in every figure; stranded is a neutral grey so it does not read as data of interest
COLORS = {"operational": "#0072b2", "excursion": "#cc79a7", "drift": "#e69f00", "ascent": "#d55e00", "stranded": "#6b6a66"}
OP_ALT, BAND = 1068.5, 15.0


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=8.5)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def daily_states(states: pd.DataFrame) -> pd.DataFrame:
    st = states.assign(EPOCH=states["EPOCH"].dt.tz_convert(None))
    days = pd.date_range(st["EPOCH"].min().floor("D"), st["EPOCH"].max().floor("D"), freq="D")
    rows = []
    for _, g in st.groupby("NORAD_CAT_ID"):
        s = g.set_index("EPOCH")["STATE"]
        s = s[~s.index.duplicated()].reindex(s.index.union(days)).ffill().reindex(days)
        rows.append(s)
    return pd.concat(rows, axis=1).apply(lambda r: r.value_counts(), axis=1).fillna(0).reindex(columns=STATE_ORDER, fill_value=0)


def fig1(states):
    ds = daily_states(states)
    fig, ax = plt.subplots(figsize=(9.6, 4.6), facecolor=SURFACE)
    style(ax)
    ax.stackplot(ds.index, *[ds[c] for c in STATE_ORDER], colors=[COLORS[c] for c in STATE_ORDER], linewidth=0)
    first = states.groupby("LAUNCH")["EPOCH"].min().dt.tz_convert(None).sort_values()
    for lid, t in first.items():
        ax.plot([t, t], [262, 272], color=INK2, lw=1.0, clip_on=False)
    ax.set_ylim(0, 260)
    ax.spines['left'].set_bounds(0, 250)
    ax.set_xlim(ds.index[0], ds.index[-1])
    ax.set_ylabel("Satellites", color=INK2, fontsize=9)
    ax.set_title("Qianfan build-out by orbital-transition state (Space-Track element sets)", fontsize=10, color=INK, loc="left", pad=22)
    handles = [Line2D([0], [0], color=COLORS[c], lw=6, label=c) for c in STATE_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, fontsize=8.5, ncol=1)
    fig.savefig(OUT / "fig1_buildout.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    return ds


LAUNCHES = [("2024-140", "first launch"), ("2024-185", "most satellites stranded"), ("2025-016", "all arrived"), ("2026-104", "later launch")]
TABLE2 = {"2024-140": (18, 17, 1, 125), "2024-185": (18, 3, 15, 173), "2025-016": (18, 18, 0, 70), "2026-104": (18, 18, 0, 59)}


def fig2(states):
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 6.6), sharey=True, facecolor=SURFACE)
    for ax, (lid, note) in zip(axes.ravel(), LAUNCHES):
        style(ax)
        g = states[states["LAUNCH"] == lid]
        t0 = g["EPOCH"].min()
        x = (g["EPOCH"] - t0).dt.total_seconds() / 86400
        ax.axhspan(OP_ALT - BAND, OP_ALT + BAND, color=COLORS["operational"], alpha=0.10, lw=0)
        for s in ["drift", "ascent", "stranded", "excursion", "operational"]:
            m = (g["STATE"] == s).to_numpy()
            ax.scatter(x[m], g.loc[m, "ALT_KM"], s=2.2, color=COLORS[s], lw=0, rasterized=True)
        n, arr, strd, med = TABLE2[lid]
        ax.set_title(f"{lid}: {note}\n{n} satellites, {arr} arrived, {strd} stranded; median {med} d to arrival", fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, x.max() + 5)
    for ax in axes[1]:
        ax.set_xlabel("Days since the launch's first element set", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Mean altitude (km)", color=INK2, fontsize=9)
    axes[0, 0].set_ylim(780, 1095)
    handles = [Line2D([0], [0], marker="o", ls="", color=COLORS[c], ms=6, label=c) for c in STATE_ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS["operational"], alpha=0.10, label=f"operational band (1,068.5 ± {BAND:.0f} km)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "fig2_altitude_states.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def fig3(R):
    bins = ["(0, 6]", "(6, 12]", "(12, 24]", "(24, 48]", "(48, 96]", "(96, 168]"]
    labels = ["0–6", "6–12", "12–24", "24–48", "48–96", "96–168"]
    states = ["operational", "drift", "ascent", "excursion"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), sharey=True, facecolor=SURFACE)
    for ax, (col, title) in zip(axes, (("dr_p50", "(a) Median"), ("dr_p95", "(b) 95th percentile"))):
        style(ax)
        x = np.arange(len(bins))
        for s in states:
            d = R[R["STATE"] == s].set_index("age_bin").reindex(bins)
            few = d["n"] < 400
            ax.plot(x, d[col], "-", color=COLORS[s], lw=1.3, alpha=0.8)
            ax.plot(x[~few.to_numpy()], d.loc[~few, col], "o", color=COLORS[s], ms=5.5, label=s)
            if few.any():
                ax.plot(x[few.to_numpy()], d.loc[few, col], "o", mfc=SURFACE, mec=COLORS[s], ms=5.5, mew=1.3)
        ax.axhline(10, color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.text(-0.25, 11.5, "10 km", ha="left", fontsize=8, color=INK2)
        ax.set_yscale("log"); ax.set_ylim(0.01, 3e4); ax.minorticks_off()
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_xlabel("Propagation age (h)", color=INK2, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    axes[0].set_ylabel("|position residual| (km)", color=INK2, fontsize=9)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.savefig(OUT / "fig3_residuals.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")


def main():
    states = load_states()
    R = pd.read_csv(OUT / "residual_summary.csv")
    ds = fig1(states)
    fig2(states)
    fig3(R)
    last = ds.iloc[-1].astype(int)
    lines = ["# Figures 1-3 numbers (30_data_figs.py)", "",
             f"- Fig. 1: {states['NORAD_CAT_ID'].nunique()} satellites, {states['LAUNCH'].nunique()} launches, {states['EPOCH'].min():%Y-%m-%d} to {states['EPOCH'].max():%Y-%m-%d}; "
             f"on the last day {int(last.sum())} satellites: " + ", ".join(f"{k} {v}" for k, v in last.items()),
             "- Fig. 2: launches " + ", ".join(l for l, _ in LAUNCHES) + " (Table 2 values in the panel titles)",
             "- Fig. 3: operational 95th percentile at 24-48 h and 48-96 h: "
             + ", ".join(f"{R[(R.STATE == 'operational') & (R.age_bin == b)]['dr_p95'].iloc[0]}" for b in ("(24, 48]", "(48, 96]")) + " km"]
    (ROOT / "FIG123_NUMBERS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
