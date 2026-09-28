"""Pilot figures and headline numbers from the outputs of 02-05.

    superseded/pilot_fig1_buildout.png       satellites per state over time (manuscript Fig. 1 is drawn by 30_data_figs.py)
    superseded/pilot_fig3_polar_pairs.png    close Qianfan-Qianfan pairs per screening epoch vs operational population
    superseded/pilot_fig4_corridor.png       altitude corridor: stranded Qianfan + CZ-6A fragments vs the ascent path
    superseded/pilot_fig5_persistence.png    day-by-day d_min of the closest operational pairs (daily screen)
    (pilot diagnostics only; none of these is a manuscript figure)
    pilot_numbers.json      numbers quoted in the findings

Usage:  py 06_figures.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, load_gp, load_states

STATE_ORDER = ["drift", "ascent", "operational", "excursion", "stranded"]
COLORS = {"drift": "#e69f00", "ascent": "#d55e00", "operational": "#0072b2",
          "excursion": "#cc79a7", "stranded": "#000000"}


def daily_states(states: pd.DataFrame) -> pd.DataFrame:
    states = states.assign(EPOCH=states["EPOCH"].dt.tz_convert(None))
    days = pd.date_range(states["EPOCH"].min().floor("D"), states["EPOCH"].max().floor("D"), freq="D")
    rows = []
    for _, g in states.groupby("NORAD_CAT_ID"):
        s = g.set_index("EPOCH")["STATE"]
        s = s[~s.index.duplicated()].reindex(s.index.union(days)).ffill().reindex(days)
        rows.append(s)
    return pd.concat(rows, axis=1).apply(lambda r: r.value_counts(), axis=1).fillna(0).reindex(columns=STATE_ORDER, fill_value=0)


def main():
    states = load_states()
    nums = {}

    # ---- fig 1: build-out ----------------------------------------------------------------
    ds = daily_states(states)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.stackplot(ds.index, *[ds[c] for c in STATE_ORDER], labels=STATE_ORDER, colors=[COLORS[c] for c in STATE_ORDER])
    ax.set(ylabel="satellites", title="Qianfan build-out by orbital-transition state (Space-Track element sets)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "superseded" / "pilot_fig1_buildout.png", dpi=150); plt.close(fig)
    nums["latest_state_counts"] = ds.iloc[-1].astype(int).to_dict()

    # ---- fig 3: polar pairs vs population ------------------------------------------------
    p14 = pd.read_csv(OUT / "encounter_pairs_14d.csv", parse_dates=["t0"])
    qq = p14[(p14["pair_type"] == "QQ_diff_launch") & (p14["STATE_1"] == "operational") & (p14["STATE_2"] == "operational")]
    ops = ds["operational"].reindex(pd.DatetimeIndex(sorted(p14["t0"].dt.tz_convert(None).unique()))).fillna(0)
    per = pd.DataFrame({f"le{t}": qq[qq["d_min_km"] <= t].groupby(qq["t0"].dt.tz_convert(None)).size() for t in (5, 10, 20)})
    per = per.reindex(ops.index).fillna(0)
    per["n_op"] = ops.values
    per["inter_pairs"] = per["n_op"] * (per["n_op"] - 1) / 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for t, c in zip((5, 10, 20), ("#d55e00", "#e69f00", "#0072b2")):
        axes[0].plot(per.index, per[f"le{t}"], "o-", ms=3, color=c, label=f"d_min <= {t} km")
        m = per["inter_pairs"] > 0
        axes[1].plot(per.loc[m, "n_op"], 1000 * per.loc[m, f"le{t}"] / per.loc[m, "inter_pairs"], "o", ms=4, color=c, label=f"<= {t} km")
    axes[0].set(ylabel="distinct operational pairs per 24 h", title="Operational-operational close pairs")
    axes[1].set(xlabel="operational satellites", ylabel="close pairs per 1000 possible pairs",
                title="Normalised by N(N-1)/2")
    for a in axes: a.legend(fontsize=8)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "superseded" / "pilot_fig3_polar_pairs.png", dpi=150); plt.close(fig)
    qq5 = qq[qq["d_min_km"] <= 5]
    nums["op_pairs_le5km_latest_epoch"] = int(per["le5"].iloc[-1])
    nums["op_pairs_abs_lat_median_deg"] = float(qq5["abs_lat_deg"].median()) if len(qq5) else None
    nums["op_pairs_vrel_median_kms"] = float(qq5["v_rel_kms"].median()) if len(qq5) else None
    per.to_csv(OUT / "polar_pairs_per_epoch.csv")

    # ---- fig 4: corridor -----------------------------------------------------------------
    cz = load_gp(sorted((DATA / "gp_history_cz6a").glob("*.json.gz")))
    last = cz["EPOCH"].max()
    cz_now = cz[cz["EPOCH"] >= last - pd.Timedelta(days=5)].groupby("NORAD_CAT_ID").last()
    cz_start = cz[cz["EPOCH"] <= cz["EPOCH"].min() + pd.Timedelta(days=30)].groupby("NORAD_CAT_ID").first()
    st_last = states.groupby("NORAD_CAT_ID").last()
    stranded = st_last[st_last["STATE"] == "stranded"]
    asc = states[states["STATE"] == "ascent"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(600, 1200, 10)
    ax.hist(cz_start["ALT_KM"], bins=bins, alpha=0.35, color="grey", label=f"CZ-6A fragments, first month (n={len(cz_start)})")
    ax.hist(cz_now["ALT_KM"], bins=bins, alpha=0.8, color="#cc79a7", label=f"CZ-6A fragments, now (n={len(cz_now)})")
    ax.hist(stranded["ALT_KM"], bins=bins, color="black", label=f"stranded Qianfan (n={len(stranded)})")
    ax2 = ax.twinx()
    ax2.hist(asc["ALT_KM"], bins=bins, histtype="step", color="#d55e00", lw=1.5, density=True, label="ascent element sets (density)")
    ax2.set_yticks([])
    ax.axvline(states.loc[states["STATE"] == "operational", "ALT_KM"].median(), color="#0072b2", ls="--", label="operational shell")
    ax.set(xlabel="mean altitude (km)", ylabel="objects", title="The deployment corridor: passive objects every new satellite must climb through")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "superseded" / "pilot_fig4_corridor.png", dpi=150); plt.close(fig)
    nums["cz6a_in_orbit_now"] = int(len(cz_now))
    nums["cz6a_now_alt_p10_p50_p90"] = [float(x) for x in cz_now["ALT_KM"].quantile([.1, .5, .9]).round(0)]
    nums["stranded_n"] = int(len(stranded))
    nums["stranded_alt_min_max"] = [float(stranded["ALT_KM"].min().round(0)), float(stranded["ALT_KM"].max().round(0))]
    cz_pairs = p14[p14["pair_type"] == "QB_CZ6A"]
    nums["cz6a_pairs_le10km_by_state"] = cz_pairs[cz_pairs["d_min_km"] <= 10].groupby("STATE_1").size().to_dict()

    # ---- fig 5: persistence --------------------------------------------------------------
    pd_ = pd.read_csv(OUT / "encounter_pairs_daily.csv", parse_dates=["t0"])
    op = pd_[(pd_["pair_type"] == "QQ_diff_launch") & (pd_["STATE_1"] == "operational") & (pd_["STATE_2"] == "operational")]
    op = op.assign(pair=op["OBJECT_ID_1"] + " / " + op["OBJECT_ID_2"], day=op["t0"].dt.tz_convert(None).dt.date)
    close = op.groupby("pair")["d_min_km"].min().nsmallest(15).index
    mat = op[op["pair"].isin(close)].pivot_table(index="pair", columns="day", values="d_min_km", aggfunc="min")
    mat = mat.loc[mat.min(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(mat.values, aspect="auto", cmap="magma", vmin=0, vmax=20)
    ax.set_yticks(range(len(mat)), mat.index, fontsize=7)
    ax.set_xticks(range(0, mat.shape[1], 3), [str(d) for d in mat.columns[::3]], rotation=45, fontsize=7)
    fig.colorbar(im, ax=ax, label="d_min in 24 h (km); blank = no pass <= 20 km")
    ax.set_title("Persistence of the closest operational polar-crossing pairs (daily screens)")
    fig.tight_layout(); fig.savefig(OUT / "superseded" / "pilot_fig5_persistence.png", dpi=150); plt.close(fig)
    days = op["day"].nunique()
    pres = op[op["d_min_km"] <= 5].groupby("pair")["day"].nunique()
    nums["daily_days"] = int(days)
    nums["pairs_le5km_any_day"] = int(len(pres))
    nums["pairs_le5km_on_ge80pct_days"] = int((pres >= 0.8 * days).sum())
    nums["pairs_le5km_days_present_median"] = float(pres.median()) if len(pres) else None
    nums["pairs_le5km_on_ge7_days"] = int((pres >= 7).sum())
    nums["pairs_le5km_days_present_max"] = int(pres.max()) if len(pres) else None
    # growth relative to the N^2 baseline: close pairs per 1000 possible pairs, early vs late
    m = per["inter_pairs"] > 0
    early, late = per[m & (per.index < "2026-04-01")], per[m & (per.index >= "2026-04-01")]
    for t in (5, 10, 20):
        nums[f"per1000_le{t}_before_2026-04"] = round(1000 * early[f"le{t}"].sum() / early["inter_pairs"].sum(), 3)
        nums[f"per1000_le{t}_from_2026-04"] = round(1000 * late[f"le{t}"].sum() / late["inter_pairs"].sum(), 3)

    (OUT / "pilot_numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    print(json.dumps(nums, indent=2, default=str))


if __name__ == "__main__":
    main()
