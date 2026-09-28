"""Per-crossing-angle and per-margin normalisation of close-approach frequency.

For each of the screening windows in encounter_pairs_3d.csv (24 h centred on t0 + 12 h) every operational
inter-plane pair from different launches is an "at-risk pair-window". For each one we compute, from that
window's element sets: crossing angle gamma, the nominal margin (model distance with both slot errors
removed), the model distance now and its 24 h window minimum (phase drift included), slot errors and
relative drift. Observed events are the SGP4-screen pairs within 5 / 10 km in that window.

Tables:
  A  rate per 1000 pair-windows by crossing-angle multiple k (gamma ~ k * 20.5 deg): observed vs model
  B  rate by nominal margin bin: observed vs model vs a Gaussian slot-error expectation
  C  rate by quarter, and the model's own prediction, to separate composition from behaviour

Usage:  py 11_per_angle.py
"""
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from common import DATA, OUT, load_gp, load_states

_spec = importlib.util.spec_from_file_location("rb", OUT.parent / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
cm = rb.cm

SPACING = 20.5     # deg between adjacent filled planes (RAAN spacing seen in 05_planes)
THR = 10.0


def window_arrays(el, iu, ju):
    """Model minimum distance over +-12 h with linear phase drift, vectorised over pairs."""
    a, b = el.iloc[iu].reset_index(drop=True), el.iloc[ju].reset_index(drop=True)
    h1, h2 = cm.unit_h(a["raan"].to_numpy(), a["inc"].to_numpy()), cm.unit_h(b["raan"].to_numpy(), b["inc"].to_numpy())
    line = np.cross(h1, h2)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    gamma = np.degrees(np.arccos(np.clip((h1 * h2).sum(1), -1, 1)))
    ea, wa, eb, wb = (a["ECCENTRICITY"].to_numpy(), a["ARG_OF_PERICENTER"].to_numpy(),
                      b["ECCENTRICITY"].to_numpy(), b["ARG_OF_PERICENTER"].to_numpy())
    n_a = np.sqrt(cm.MU / a["A_KM"].to_numpy() ** 3) * 86400
    n_b = np.sqrt(cm.MU / b["A_KM"].to_numpy() ** 3) * 86400
    best = np.full(len(a), np.inf)
    drift = np.zeros(len(a))
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
        d = np.hypot(xmin, r1 - r2)
        upd = d < best
        best = np.where(upd, d, best)
        drift = np.where(upd, np.abs(sl), drift)
    return gamma, best, drift


def relabel(pairs: pd.DataFrame, df: pd.DataFrame, states: pd.DataFrame, max_age_h: float = 24) -> pd.DataFrame:
    """Replace STATE_1/STATE_2 of screened pairs with the labels of another state classification.

    The screen used, for each satellite, the element set nearest to t0 within max_age_h (04_screen.nearest);
    the same element set is looked up here, so the baseline labels are reproduced exactly.
    """
    lab = df[["NORAD_CAT_ID", "EPOCH"]].merge(states[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"])
    lab = lab.sort_values("EPOCH")
    out = pairs.copy()
    for i in (1, 2):
        q = out[["t0", f"NORAD_CAT_ID_{i}"]].drop_duplicates().rename(columns={f"NORAD_CAT_ID_{i}": "NORAD_CAT_ID"})
        q = pd.merge_asof(q.sort_values("t0"), lab, left_on="t0", right_on="EPOCH", by="NORAD_CAT_ID",
                          direction="nearest", tolerance=pd.Timedelta(hours=max_age_h))
        q = q.rename(columns={"NORAD_CAT_ID": f"NORAD_CAT_ID_{i}", "STATE": f"STATE_{i}"})[["t0", f"NORAD_CAT_ID_{i}", f"STATE_{i}"]]
        out = out.drop(columns=f"STATE_{i}").merge(q, on=["t0", f"NORAD_CAT_ID_{i}"], how="left")
    return out


def poisson_ci(k):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="encounter_pairs_3d.csv", help="screen output file in out/")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--states-tag", default="", help="use data/states{tag}.csv (band-width sensitivity) and relabel "
                                                     "the screened pairs with it; the screen itself covers all states")
    args = ap.parse_args()
    pairs = pd.read_csv(OUT / args.pairs, parse_dates=["t0"])
    pairs["t0"] = pd.to_datetime(pairs["t0"], utc=True)
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states(args.states_tag)
    if args.states_tag:
        pairs = relabel(pairs, df, states)
    obs = pairs[(pairs["pair_type"] == "QQ_diff_launch") & (pairs["STATE_1"] == "operational") & (pairs["STATE_2"] == "operational")]
    obs = obs.assign(a=np.minimum(obs["OBJECT_ID_1"], obs["OBJECT_ID_2"]), b=np.maximum(obs["OBJECT_ID_1"], obs["OBJECT_ID_2"]))
    obs = obs.groupby(["t0", "a", "b"])["d_min_km"].min().reset_index()

    frames = []
    windows = sorted(pairs["t0"].unique())
    for n, t0 in enumerate(windows):
        t0 = pd.Timestamp(t0)
        try:
            el = rb.get_el(df, states, t0 + pd.Timedelta(hours=12))
        except ValueError:          # no operational satellites yet
            continue
        if len(el) < 6:
            continue
        el = rb.slots(el, rb.ESTIMATORS["circular mean"])
        iu, ju = np.triu_indices(len(el), 1)
        keep = (el["plane"].to_numpy()[iu] != el["plane"].to_numpy()[ju]) & (el["LAUNCH"].to_numpy()[iu] != el["LAUNCH"].to_numpy()[ju])
        iu, ju = iu[keep], ju[keep]
        if not len(iu):
            continue
        gamma, dwin, drift = window_arrays(el, iu, ju)
        nom = cm.model_mean(el, iu, ju, "ub_slotted")["d_model_km"].to_numpy()
        now = cm.model_mean(el, iu, ju, "ub")["d_model_km"].to_numpy()
        ids = el["OBJECT_ID"].to_numpy()
        slot = el["slot_err_deg"].to_numpy()
        f = pd.DataFrame({"t0": t0, "a": np.minimum(ids[iu], ids[ju]), "b": np.maximum(ids[iu], ids[ju]), "gamma": gamma,
                          "d_win": dwin, "d_now": now, "nominal": nom, "drift": drift,
                          "slot_max": np.maximum(np.abs(slot[iu]), np.abs(slot[ju])), "slot_rel": slot[iu] - slot[ju],
                          "n_op": len(el), "n_planes": el["plane"].nunique()})
        frames.append(f)
        if (n + 1) % 50 == 0:
            print(f"  built {n + 1}/{len(windows)} windows", flush=True)
    P = pd.concat(frames, ignore_index=True)
    P = P.merge(obs.rename(columns={"d_min_km": "d_obs"}), on=["t0", "a", "b"], how="left")
    P["obs10"] = P["d_obs"] <= THR
    P["obs5"] = P["d_obs"] <= 5
    P["mod10"] = P["d_win"] <= THR
    P["k"] = np.round(P["gamma"] / SPACING).astype(int)
    P["std_angle"] = (P["gamma"] - P["k"] * SPACING).abs() < 2.0
    P["cosg2"] = np.cos(np.radians(P["gamma"]) / 2)
    P["good"] = (P["slot_max"] <= 1.0) & (P["drift"] < 50)
    P["quarter"] = P["t0"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    print(f"{len(P):,} pair-windows in {P['t0'].nunique()} windows; observed <= {THR:g} km: {P['obs10'].sum()}, "
          f"<= 5 km: {P['obs5'].sum()}; model window-min <= {THR:g} km: {P['mod10'].sum()}")
    print(f"pairs with off-grid or fast-drifting satellite: {(~P['good']).mean():.0%} of pair-windows")

    # sanity: does the model agree with the screen pair-by-pair?
    both = P[P["mod10"] | P["obs10"]]
    print(f"agreement of model (window min <= 10 km) and SGP4 screen: both {int((both['mod10'] & both['obs10']).sum())}, "
          f"model only {int((both['mod10'] & ~both['obs10']).sum())}, screen only {int((~both['mod10'] & both['obs10']).sum())}")
    g = P[P["good"] & (P["mod10"] | P["obs10"])]
    print(f"   restricted to on-grid, slow-drifting pairs: both {int((g['mod10'] & g['obs10']).sum())}, model only "
          f"{int((g['mod10'] & ~g['obs10']).sum())}, screen only {int((~g['mod10'] & g['obs10']).sum())}")

    def rate_table(D, by, label):
        rows = []
        for key, g in D.groupby(by):
            n, o, m = len(g), int(g["obs10"].sum()), int(g["mod10"].sum())
            lo, hi = poisson_ci(o)
            rows.append({label: key, "pair_windows": n, "observed": o, "obs_per_1000": 1000 * o / n,
                         "ci_lo": 1000 * lo / n, "ci_hi": 1000 * hi / n, "model": m, "model_per_1000": 1000 * m / n,
                         "obs/model": o / m if m else np.nan, "cos(g/2)": g["cosg2"].mean(),
                         "obs_x_cos": 1000 * o / n * g["cosg2"].mean()})
        return pd.DataFrame(rows).round(2)

    std = P[P["std_angle"]]
    print("\nA. rate of <= 10 km per 1000 pair-windows by crossing-angle multiple k (gamma ~ k*20.5 deg)")
    print("   all pairs at standard angles:")
    A_all = rate_table(std, "k", "k")
    print(A_all.to_string(index=False))
    print("   on-grid, slowly drifting pairs only:")
    A_good = rate_table(std[std["good"]], "k", "k")
    print(A_good.to_string(index=False))
    A_all.to_csv(OUT / f"per_angle_all{args.tag}.csv", index=False)
    A_good.to_csv(OUT / f"per_angle_good{args.tag}.csv", index=False)

    # heterogeneity between angles (on-grid subset): Poisson chi-square against a common rate
    tot = A_good["observed"].sum() / A_good["pair_windows"].sum()
    exp = tot * A_good["pair_windows"]
    keep = exp >= 5
    stat = (((A_good["observed"] - exp) ** 2 / exp)[keep]).sum()
    print(f"   heterogeneity across k (on-grid subset, {int(keep.sum())} bins with expected >= 5): chi2 = {stat:.1f}, "
          f"dof = {int(keep.sum()) - 1}, p = {chi2.sf(stat, int(keep.sum()) - 1):.2g}")

    # ---------- B: by nominal margin, with a Gaussian slot-error expectation
    G = P[P["good"] & P["std_angle"]].copy()
    sig_deg = 1.4826 * np.median(np.abs(G["slot_rel"] - np.median(G["slot_rel"])))
    G["sigma_x"] = 7446 * np.radians(sig_deg) * G["cosg2"]
    G["p_gauss"] = norm.cdf((THR - G["nominal"]) / G["sigma_x"]) - norm.cdf((-THR - G["nominal"]) / G["sigma_x"])
    edges = [0, 5, 10, 15, 20, 30, 40, 60, 100, 200, 1e9]
    G["margin_bin"] = pd.cut(G["nominal"], edges)
    rows = []
    for key, g in G.groupby("margin_bin", observed=True):
        o, n = int(g["obs10"].sum()), len(g)
        lo, hi = poisson_ci(o)
        rows.append({"nominal_margin_km": str(key), "pair_windows": n, "observed": o, "obs_per_1000": 1000 * o / n,
                     "ci_lo": 1000 * lo / n, "ci_hi": 1000 * hi / n, "model_per_1000": 1000 * g["mod10"].mean(),
                     "gauss_per_1000": 1000 * g["p_gauss"].mean()})
    B = pd.DataFrame(rows).round(2)
    print(f"\nB. by nominal margin (both slot errors removed), on-grid slow pairs at standard angles; relative slot-error sigma = {sig_deg:.2f} deg")
    print(B.to_string(index=False))
    B.to_csv(OUT / f"per_margin{args.tag}.csv", index=False)

    # ---------- nominal-margin structure by plane-step k (why some angles are safe)
    print("\nA2. nominal margin (slot errors removed) by plane step k, on-grid slow pairs at standard angles")
    rows = []
    for k, g in G.groupby("k"):
        near = g["nominal"] < 60
        rows.append({"k": k, "pair_windows": len(g), "share_nominal<60km": near.mean(), "share_nominal<20km": (g["nominal"] < 20).mean(),
                     "median_nominal_km": g["nominal"].median(), "events_from_nominal<60": int((g["obs10"] & near).sum()),
                     "events_from_nominal>=60": int((g["obs10"] & ~near).sum())})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    e_near = int((G["obs10"] & (G["nominal"] < 60)).sum()); e_all = int(G["obs10"].sum())
    print(f"   pair-windows with nominal margin < 60 km: {(G['nominal'] < 60).mean():.2%} of at-risk pair-windows, "
          f"carrying {e_near}/{e_all} = {e_near / e_all:.0%} of the observed <= {THR:g} km events")
    P.to_pickle(DATA / f"pair_windows{args.tag}.pkl.gz")

    # ---------- C: by quarter
    print("\nC. by quarter (on-grid, slowly drifting, standard angles): observed vs model and composition")
    rows = []
    for q, g in G.groupby("quarter"):
        o, n = int(g["obs10"].sum()), len(g)
        rows.append({"quarter": q, "pair_windows": n, "observed": o, "obs_per_1000": 1000 * o / n,
                     "model_per_1000": 1000 * g["mod10"].mean(), "gauss_per_1000": 1000 * g["p_gauss"].mean(),
                     "median_nominal_km": g["nominal"].median(), "share_nominal<10": (g["nominal"] < 10).mean(),
                     "median_planes": g["n_planes"].median()})
    C = pd.DataFrame(rows).round(2)
    print(C.to_string(index=False))
    C.to_csv(OUT / f"per_quarter{args.tag}.csv", index=False)
    print("\n   all operational inter-plane pair-windows (no on-grid/drift restriction):")
    rows = []
    for q, g in P.groupby("quarter"):
        rows.append({"quarter": q, "pair_windows": len(g), "obs_per_1000": 1000 * g["obs10"].mean(), "model_per_1000": 1000 * g["mod10"].mean(),
                     "share_good": g["good"].mean(), "share_drift>=50": (g["drift"] >= 50).mean(), "median_planes": g["n_planes"].median()})
    print(pd.DataFrame(rows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
