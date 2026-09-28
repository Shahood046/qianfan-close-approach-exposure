"""Episode statistics for operational Qianfan-Qianfan polar close approaches, 2024-2026.

Input: out/encounter_pairs_3d.csv from `04_screen.py --every-days 3 --kinds QQ --tag _3d`.

An episode is a run of consecutive screening windows in which a given pair of operational satellites
is within --thr km (default 10) on each window, allowing --gap missed windows. Windows are 24 h every
3 days, so durations are lower bounds with a resolution of 3 days, and episodes shorter than ~3 days can
be missed. The counting unit is the episode, never the pass or the day.

Answers, in order:
  Q1 how many distinct episodes                   Q5 minimum separation vs slot error (see below)
  Q2 how long they last                           Q6 model accuracy over the whole history
  Q3 how small they get                           Q7 how episodes change as planes are populated
  Q4 which inter-plane geometries produce them

Q5-Q6 evaluate the crossing model at each episode's closest window: element sets of that window's
epoch give the model d and the slot errors of both satellites (relative to their plane's mean phase).

Usage:  py 10_episodes.py [--thr 10] [--gap 1]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, load_gp, load_states
import importlib.util

_spec = importlib.util.spec_from_file_location("rb", OUT.parent / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
cm = rb.cm

STEP_D = 3.0


def episodes(pairs: pd.DataFrame, thr: float, gap: int) -> pd.DataFrame:
    op = pairs[(pairs["pair_type"] == "QQ_diff_launch") & (pairs["STATE_1"] == "operational")
               & (pairs["STATE_2"] == "operational") & (pairs["d_min_km"] <= thr)].copy()
    op["t0"] = pd.to_datetime(op["t0"], utc=True)
    all_windows = np.sort(pairs["t0"].pipe(pd.to_datetime, utc=True).unique())
    widx = {t: i for i, t in enumerate(all_windows)}
    op["w"] = op["t0"].map(widx)
    rows = []
    for (a, b), g in op.groupby(["OBJECT_ID_1", "OBJECT_ID_2"]):
        g = g.sort_values("w")
        start = prev = None
        members = []
        for _, r in g.iterrows():
            if start is None or r["w"] - prev > gap + 1:
                if members:
                    rows.append(_summ(a, b, members))
                members = []
            members.append(r)
            start = start if members and len(members) > 1 else r["w"]
            prev = r["w"]
        if members:
            rows.append(_summ(a, b, members))
    return pd.DataFrame(rows)


def _summ(a, b, m):
    g = pd.DataFrame(m)
    i = g["d_min_km"].idxmin()
    return dict(id_1=a, id_2=b, start=g["t0"].min(), end=g["t0"].max(),
                n_windows=len(g), duration_d_min=(g["t0"].max() - g["t0"].min()).days + STEP_D,
                dmin_km=g["d_min_km"].min(), t_dmin=g.loc[i, "t0"], vrel_kms=g["v_rel_kms"].median(),
                abs_lat=g["abs_lat_deg"].median())


def model_window(el, i, j):
    """Model minimum distance over a 24 h window centred on the element epoch.

    The phase difference at the crossing drifts linearly at (n_a - n_b); the distance is
    sqrt((r*dphi(t)*cos(gamma/2))^2 + dr^2) minimised over t in [-12 h, +12 h] and both crossings.
    """
    a, b = el.iloc[i].reset_index(drop=True), el.iloc[j].reset_index(drop=True)
    h1, h2 = cm.unit_h(a["raan"].to_numpy(), a["inc"].to_numpy()), cm.unit_h(b["raan"].to_numpy(), b["inc"].to_numpy())
    line = np.cross(h1, h2)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    gamma = np.degrees(np.arccos(np.clip((h1 * h2).sum(1), -1, 1)))
    ea, wa, eb, wb = (a["ECCENTRICITY"].to_numpy(), a["ARG_OF_PERICENTER"].to_numpy(),
                      b["ECCENTRICITY"].to_numpy(), b["ARG_OF_PERICENTER"].to_numpy())
    n_a = np.sqrt(cm.MU / a["A_KM"].to_numpy() ** 3) * 86400      # rad/day
    n_b = np.sqrt(cm.MU / b["A_KM"].to_numpy() ** 3) * 86400
    best = np.full(len(a), np.inf)
    for sgn in (1, -1):
        p = sgn * line
        us1 = cm.u_of_direction(p, a["raan"].to_numpy(), a["inc"].to_numpy())
        us2 = cm.u_of_direction(p, b["raan"].to_numpy(), b["inc"].to_numpy())
        dphi = np.radians(cm.wrap((a["ub"].to_numpy() - cm.mean_phase(us1, ea, wa)) - (b["ub"].to_numpy() - cm.mean_phase(us2, eb, wb))))
        r1 = a["A_KM"].to_numpy() * (1 - ea * np.cos(np.radians(us1 - wa)))
        r2 = b["A_KM"].to_numpy() * (1 - eb * np.cos(np.radians(us2 - wb)))
        k = 0.5 * (r1 + r2) * np.cos(np.radians(gamma) / 2)
        x0, sl = k * dphi, k * (n_a - n_b)
        lo, hi = x0 - 0.5 * np.abs(sl), x0 + 0.5 * np.abs(sl)
        xmin = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(np.abs(lo), np.abs(hi)))
        best = np.minimum(best, np.hypot(xmin, r1 - r2))
    return float(best[0]), float(np.abs(sl[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thr", type=float, default=10)
    ap.add_argument("--gap", type=int, default=1, help="windows that may be missed inside an episode")
    args = ap.parse_args()

    pairs = pd.read_csv(OUT / "encounter_pairs_3d.csv", parse_dates=["t0"])
    pairs["t0"] = pd.to_datetime(pairs["t0"], utc=True)
    n_win = pairs["t0"].nunique()
    ep = episodes(pairs, args.thr, args.gap)
    ep.to_csv(OUT / "episodes.csv", index=False)
    print(f"{n_win} screening windows; operational QQ pairs within {args.thr:g} km: episodes = {len(ep)}")
    if ep.empty:
        return

    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    idx_cache = {}

    # ---- per-episode geometry: plane pair, gamma, model d and slot errors at the closest window
    geo = []
    for _, e in ep.iterrows():
        t = e["t_dmin"] + pd.Timedelta(hours=12)
        key = t.floor("D")
        if key not in idx_cache:
            el = rb.slots(rb.get_el(df, states, t), rb.ESTIMATORS["circular mean"])
            idx_cache = {key: (el, {o: i for i, o in enumerate(el["OBJECT_ID"])})}
        el, idx = idx_cache[key]
        if e["id_1"] not in idx or e["id_2"] not in idx:
            geo.append(dict(plane_1=np.nan, plane_2=np.nan, gamma=np.nan, d_model=np.nan, d_window=np.nan, d_slotted=np.nan,
                            drift_kmd=np.nan, slot_1=np.nan, slot_2=np.nan, n_planes=el["plane"].nunique(), n_op=len(el)))
            continue
        i, j = np.array([idx[e["id_1"]]]), np.array([idx[e["id_2"]]])
        m = cm.model_mean(el, i, j, "ub")
        dw, drift = model_window(el, i, j)
        ds = cm.model_mean(el, i, j, "ub_slotted")["d_model_km"].iloc[0]
        inter = int(sum(1 for _ in range(1)) and (el["plane"].to_numpy()[:, None] != el["plane"].to_numpy()[None, :]).sum() // 2)
        geo.append(dict(plane_1=el["plane"].iloc[i[0]], plane_2=el["plane"].iloc[j[0]], gamma=m["gamma_deg"].iloc[0],
                        d_model=m["d_model_km"].iloc[0], d_window=dw, d_slotted=ds, drift_kmd=drift,
                        slot_1=el["slot_err_deg"].iloc[i[0]], slot_2=el["slot_err_deg"].iloc[j[0]],
                        n_planes=el["plane"].nunique(), n_op=len(el), inter_pairs=inter))
    ep = pd.concat([ep.reset_index(drop=True), pd.DataFrame(geo)], axis=1)
    ep["gamma_bin"] = ep["gamma"].round(0)
    ep.to_csv(OUT / "episodes.csv", index=False)

    q = lambda s: s.quantile([.1, .5, .9]).round(1).tolist()
    print(f"\nQ1  episodes: {len(ep)}; distinct pairs: {ep.groupby(['id_1', 'id_2']).ngroups}; "
          f"pairs with >1 episode: {(ep.groupby(['id_1', 'id_2']).size() > 1).sum()}")
    print(f"Q2  duration (days, resolution {STEP_D:g}): 10/50/90th pct {q(ep['duration_d_min'])}; "
          f"windows in episode: {q(ep['n_windows'])}; episodes spanning >=3 windows: {(ep['n_windows'] >= 3).mean():.0%}")
    print(f"Q3  minimum distance (km): 10/50/90th pct {q(ep['dmin_km'])}; < 1 km: {(ep['dmin_km'] < 1).sum()}, < 2 km: {(ep['dmin_km'] < 2).sum()}, < 5 km: {(ep['dmin_km'] < 5).sum()}")
    print("Q4  by plane-crossing angle gamma (deg):")
    g = ep.groupby("gamma_bin").agg(episodes=("dmin_km", "size"), median_dmin=("dmin_km", "median"), median_days=("duration_d_min", "median"),
                                    pairs=("id_1", lambda s: s.nunique())).round(1)
    print(g.to_string())
    print("    adjacent-plane pairs (|plane_1 - plane_2| == 1 in RAAN order) vs others:")
    ep["adjacent"] = (ep["plane_1"] - ep["plane_2"]).abs() == 1
    print(ep.groupby("adjacent").agg(episodes=("dmin_km", "size"), median_dmin=("dmin_km", "median")).round(1).to_string())

    ok = ep.dropna(subset=["d_window"]).copy()
    ok["on_grid"] = np.maximum(ok["slot_1"].abs(), ok["slot_2"].abs()) <= 1.0
    ok["err"] = (ok["d_window"] - ok["dmin_km"]).abs()
    print(f"\nQ6  model (24 h window minimum with phase drift) vs SGP4 screen, n={len(ok)} episodes:")
    for name, g in (("all", ok), ("both satellites within 1 deg of grid", ok[ok["on_grid"]]), ("a satellite still being phased (>1 deg)", ok[~ok["on_grid"]])):
        print(f"    {name:42s} n={len(g):3d}  r(log)={np.corrcoef(np.log(g['d_window'] + .2), np.log(g['dmin_km'] + .2))[0, 1]:.2f}  "
              f"median |err|={g['err'].median():.2f} km  p90={g['err'].quantile(.9):.2f} km  median drift={g['drift_kmd'].median():.1f} km/day")
    print("    on-grid episodes by year:")
    og = ok[ok["on_grid"]].assign(year=lambda d: d["t_dmin"].dt.year)
    print(og.groupby("year").agg(n=("err", "size"), median_err=("err", "median"), p90_err=("err", lambda s: s.quantile(.9))).round(2).to_string())

    og = ok[ok["on_grid"]]
    print(f"\nQ5  counterfactual for the {len(og)} on-grid episodes (slot errors removed, everything else actual):")
    print(f"    observed d_min: median {og['dmin_km'].median():.1f} km; slotted model: median {og['d_slotted'].median():.1f} km; "
          f"share > 10 km: {(og['d_slotted'] > 10).mean():.0%}; > 20 km: {(og['d_slotted'] > 20).mean():.0%}")
    close = og[og["dmin_km"] < 3]
    print(f"    the {len(close)} episodes with observed d_min < 3 km: slotted median {close['d_slotted'].median():.1f} km, "
          f"share > 10 km {(close['d_slotted'] > 10).mean():.0%}, > 20 km {(close['d_slotted'] > 20).mean():.0%}")
    print(f"    still-phasing episodes: {(~ok['on_grid']).sum()} ({(~ok['on_grid']).mean():.0%}), observed median d_min {ok.loc[~ok['on_grid'], 'dmin_km'].median():.1f} km, "
          f"median duration {ok.loc[~ok['on_grid'], 'duration_d_min'].median():.0f} d vs on-grid {og['duration_d_min'].median():.0f} d")
    ep["on_grid"] = ep.index.map(ok["on_grid"])
    ep.to_csv(OUT / "episodes.csv", index=False)

    # ---- Q7 longitudinal: episodes per possible operational pair as planes fill
    ops = load_states()
    ops = ops[ops["STATE"] == "operational"]
    n_op = (ops.assign(day=ops["EPOCH"].dt.floor("D")).groupby("day")["NORAD_CAT_ID"].nunique())
    n_op.index = n_op.index.tz_convert(None)
    ep["start_utc"] = ep["start"].dt.tz_convert(None)
    ep["quarter"] = ep["start_utc"].dt.to_period("Q").astype(str)
    q_ep = ep.groupby("quarter").size()
    q_n = n_op.groupby(n_op.index.to_period("Q").astype(str)).mean()
    q_planes = ep.groupby("quarter")["n_planes"].median()
    tab = pd.DataFrame({"episodes_started": q_ep, "mean_operational_sats": q_n.reindex(q_ep.index),
                        "median_planes": q_planes}).dropna()
    tab["possible_pairs"] = tab["mean_operational_sats"] * (tab["mean_operational_sats"] - 1) / 2
    tab["per_1000_pairs"] = 1000 * tab["episodes_started"] / tab["possible_pairs"]
    tab["mean_inter_plane_pairs"] = ep.groupby("quarter")["inter_pairs"].mean().reindex(tab.index)
    tab["per_1000_interplane"] = 1000 * tab["episodes_started"] / tab["mean_inter_plane_pairs"]
    og_q = ep[ep["on_grid"] == True].groupby("quarter").size()
    tab["on_grid_episodes"] = og_q.reindex(tab.index).fillna(0)
    tab["on_grid_per_1000_interplane"] = 1000 * tab["on_grid_episodes"] / tab["mean_inter_plane_pairs"]
    print("\nQ7  episodes started per quarter (normalised by N(N-1)/2 of operational satellites, not by inter-plane pairs):")
    print(tab.round(2).to_string())
    tab.to_csv(OUT / "episodes_by_quarter.csv")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(ep["dmin_km"], bins=np.arange(0, args.thr + 0.5, 0.5), color="#0072b2"); ax[0].set(xlabel="episode minimum distance (km)", ylabel="episodes")
    ax[1].hist(ep["duration_d_min"], bins=np.arange(0, ep["duration_d_min"].max() + 4, 3), color="#d55e00"); ax[1].set(xlabel="duration (days, lower bound)")
    ax[2].bar(range(len(tab)), tab["per_1000_pairs"], color="#009e73"); ax[2].set_xticks(range(len(tab)), tab.index, rotation=60, fontsize=7)
    ax[2].set(ylabel="episodes per 1000 possible pairs")
    fig.tight_layout(); fig.savefig(OUT / "fig7_episodes.png", dpi=150)


if __name__ == "__main__":
    main()
