"""Analytical polar-crossing model vs SGP4, and a slot-error-only counterfactual.

For two near-circular orbits of equal speed v crossing at plane angle gamma, the satellites reach
the crossing with an arc offset delta_phi (their phase difference relative to the crossing point)
and a radial offset dr (from each orbit's eccentricity vector). Treating the tracks as straight lines
near the crossing gives

    d_min ~= sqrt( (r * delta_phi * cos(gamma/2))^2 + dr^2 ),     v_rel = 2 v sin(gamma/2)

The crossing points are the two ends of the line h1 x h2. For each satellite, u* is the argument of
latitude at the crossing, and r(u*) = a (1 - e cos(u* - omega)) uses its mean elements. Of the two
crossings, the pair's minimum is the smaller one.

Counterfactual: remove each satellite's measured slot error (from 05_planes: u minus the nearest slot
of its plane's phase grid) and keep everything else (actual RAAN, inclination, a, e, omega).

Usage:  py 07_crossing_model.py [--date ...]   (uses 05_planes.py for planes and slot errors)
"""
from __future__ import annotations

import argparse
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, MU, OUT, load_gp, load_states

spec = importlib.util.spec_from_file_location("planes", OUT.parent / "05_planes.py")
planes_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planes_mod)


def unit_h(raan, inc):
    O, i = np.radians(raan), np.radians(inc)
    return np.stack([np.sin(O) * np.sin(i), -np.cos(O) * np.sin(i), np.cos(i)], -1)


def u_of_direction(p, raan, inc):
    """Argument of latitude of direction p in the orbit plane (raan, inc)."""
    O, i = np.radians(raan), np.radians(inc)
    node = np.stack([np.cos(O), np.sin(O), np.zeros_like(O)], -1)
    y = np.cross(unit_h(raan, inc), node)
    return np.degrees(np.arctan2((p * y).sum(-1), (p * node).sum(-1))) % 360


def wrap(x):
    return (x + 180) % 360 - 180


def model(el: pd.DataFrame, iu, ju, u_col: str) -> pd.DataFrame:
    a, b = el.iloc[iu].reset_index(drop=True), el.iloc[ju].reset_index(drop=True)
    h1, h2 = unit_h(a["raan"].to_numpy(), a["inc"].to_numpy()), unit_h(b["raan"].to_numpy(), b["inc"].to_numpy())
    line = np.cross(h1, h2)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    gamma = np.degrees(np.arccos(np.clip((h1 * h2).sum(1), -1, 1)))
    best = np.full(len(a), np.inf)
    for sgn in (1, -1):
        p = sgn * line
        us1 = u_of_direction(p, a["raan"].to_numpy(), a["inc"].to_numpy())
        us2 = u_of_direction(p, b["raan"].to_numpy(), b["inc"].to_numpy())
        dphi = np.radians(wrap((a[u_col].to_numpy() - us1) - (b[u_col].to_numpy() - us2)))
        r1 = a["A_KM"].to_numpy() * (1 - a["ECCENTRICITY"].to_numpy() * np.cos(np.radians(us1 - a["ARG_OF_PERICENTER"].to_numpy())))
        r2 = b["A_KM"].to_numpy() * (1 - b["ECCENTRICITY"].to_numpy() * np.cos(np.radians(us2 - b["ARG_OF_PERICENTER"].to_numpy())))
        r = (r1 + r2) / 2
        d = np.sqrt((r * dphi * np.cos(np.radians(gamma) / 2)) ** 2 + (r1 - r2) ** 2)
        best = np.minimum(best, d)
    v = np.sqrt(MU / a["A_KM"].to_numpy())
    return pd.DataFrame({"gamma_deg": gamma, "d_model_km": best, "vrel_model_kms": 2 * v * np.sin(np.radians(gamma) / 2)})


def mean_phase(u, e, w):
    """Osculating argument of latitude -> mean argument of latitude (first order in e, equation of centre)."""
    return u - np.degrees(2 * e * np.sin(np.radians(u - w)))


def model_mean(el: pd.DataFrame, iu, ju, ph_col: str) -> pd.DataFrame:
    """Crossing model with phase measured in mean argument of latitude.

    el[ph_col] must already be a mean argument of latitude (e.g. 'ub'); the crossing arguments of
    latitude are converted with the same equation-of-centre correction.
    """
    a, b = el.iloc[iu].reset_index(drop=True), el.iloc[ju].reset_index(drop=True)
    h1, h2 = unit_h(a["raan"].to_numpy(), a["inc"].to_numpy()), unit_h(b["raan"].to_numpy(), b["inc"].to_numpy())
    line = np.cross(h1, h2)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    gamma = np.degrees(np.arccos(np.clip((h1 * h2).sum(1), -1, 1)))
    ea, wa, eb, wb = (a["ECCENTRICITY"].to_numpy(), a["ARG_OF_PERICENTER"].to_numpy(),
                      b["ECCENTRICITY"].to_numpy(), b["ARG_OF_PERICENTER"].to_numpy())
    best = np.full(len(a), np.inf)
    for sgn in (1, -1):
        p = sgn * line
        us1 = u_of_direction(p, a["raan"].to_numpy(), a["inc"].to_numpy())
        us2 = u_of_direction(p, b["raan"].to_numpy(), b["inc"].to_numpy())
        dphi = np.radians(wrap((a[ph_col].to_numpy() - mean_phase(us1, ea, wa)) - (b[ph_col].to_numpy() - mean_phase(us2, eb, wb))))
        r1 = a["A_KM"].to_numpy() * (1 - ea * np.cos(np.radians(us1 - wa)))
        r2 = b["A_KM"].to_numpy() * (1 - eb * np.cos(np.radians(us2 - wb)))
        d = np.sqrt((0.5 * (r1 + r2) * dphi * np.cos(np.radians(gamma) / 2)) ** 2 + (r1 - r2) ** 2)
        best = np.minimum(best, d)
    v = np.sqrt(MU / a["A_KM"].to_numpy())
    return pd.DataFrame({"gamma_deg": gamma, "d_model_km": best, "vrel_model_kms": 2 * v * np.sin(np.radians(gamma) / 2)})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date")
    args = ap.parse_args()

    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    t = pd.Timestamp(args.date, tz="UTC") if args.date else df["EPOCH"].max().floor("h") - pd.Timedelta(days=1)
    near = df.assign(_dt=(df["EPOCH"] - t).abs())
    near = near[near["_dt"] <= pd.Timedelta(hours=48)]
    near = near.loc[near.groupby("NORAD_CAT_ID")["_dt"].idxmin()]
    near = near.merge(load_states()[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"])
    op = near[near["STATE"] == "operational"].reset_index(drop=True)

    el = planes_mod.elements_at(op, t)
    el["plane"] = planes_mod.cluster_planes(el["raan"].to_numpy(), 2.0)
    gaps = [np.median(np.diff(np.sort(g["u"].to_numpy()))) for _, g in el.groupby("plane") if len(g) > 2]
    s = 360 / round(360 / min(gaps))
    phase = el.groupby("plane")["u"].apply(lambda u: planes_mod.circmean(u % s, s))
    el["slot_err_deg"] = (el["u"] - el["plane"].map(phase) + s / 2) % s - s / 2
    el["u_slotted"] = el["u"] - el["slot_err_deg"]
    el = el.join(op[["A_KM", "ECCENTRICITY", "ARG_OF_PERICENTER"]])
    # the same analysis in mean argument of latitude (removes the equation-of-centre pattern)
    el["ub"] = mean_phase(el["u"], el["ECCENTRICITY"], el["ARG_OF_PERICENTER"])
    ph = el.groupby("plane")["ub"].apply(lambda u: planes_mod.circmean(u % s, s))
    el["slot_err_mean"] = (el["ub"] - el["plane"].map(ph) + s / 2) % s - s / 2
    el["ub_slotted"] = el["ub"] - el["slot_err_mean"]

    iu, ju = np.triu_indices(len(el), 1)
    keep = el["plane"].to_numpy()[iu] != el["plane"].to_numpy()[ju]
    iu, ju = iu[keep], ju[keep]
    actual = model_mean(el, iu, ju, "ub")
    counter = model_mean(el, iu, ju, "ub_slotted")
    counter_osc = model(el, iu, ju, "u_slotted")          # earlier version: slot errors in osculating phase
    res = pd.DataFrame({"id_1": el["OBJECT_ID"].to_numpy()[iu], "id_2": el["OBJECT_ID"].to_numpy()[ju],
                        "slot_err_1": el["slot_err_mean"].to_numpy()[iu], "slot_err_2": el["slot_err_mean"].to_numpy()[ju]})
    res = pd.concat([res, actual, counter[["d_model_km"]].rename(columns={"d_model_km": "d_slotted_km"}),
                     counter_osc[["d_model_km"]].rename(columns={"d_model_km": "d_slotted_osc_km"})], axis=1)

    obs = pd.read_csv(OUT / "ideal_vs_observed_pairs.csv")
    both = pd.concat([obs, obs.rename(columns={"id_1": "id_2", "id_2": "id_1"})])
    res = res.merge(both[["id_1", "id_2", "observed_dmin_km"]], on=["id_1", "id_2"], how="left")
    res.to_csv(OUT / "crossing_model_pairs.csv", index=False)

    close = res[res["observed_dmin_km"] <= 50].dropna()
    err = close["d_model_km"] - close["observed_dmin_km"]
    rho = np.corrcoef(close["d_model_km"], close["observed_dmin_km"])[0, 1]
    print(f"epoch {t:%Y-%m-%d %H:%M}; {len(el)} operational satellites, {len(res)} inter-plane pairs")
    print(f"model vs SGP4 for pairs with observed d_min <= 50 km (n={len(close)}): r = {rho:.3f}, "
          f"median |error| {err.abs().median():.2f} km, 90th pct {err.abs().quantile(0.9):.2f} km")
    print("\npairs within threshold:           <=5  <=10  <=20 km")
    for name, col in (("SGP4 observed", "observed_dmin_km"), ("model, actual slots", "d_model_km"),
                      ("model, slot errors removed", "d_slotted_km")):
        print(f"  {name:30s} " + "  ".join(f"{(res[col] <= k).sum():4d}" for k in (5, 10, 20)))
    top = res.nsmallest(10, "observed_dmin_km")[["id_1", "id_2", "gamma_deg", "slot_err_1", "slot_err_2",
                                                 "observed_dmin_km", "d_model_km", "d_slotted_km"]]
    print("\nclosest pairs:\n" + top.round(2).to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.loglog(close["observed_dmin_km"], close["d_model_km"], "o", ms=3, alpha=0.6)
    ax.plot([0.1, 60], [0.1, 60], "k--", lw=0.8)
    ax.set(xlabel="SGP4 minimum distance over one orbit (km)", ylabel="analytical crossing model (km)",
           title=f"Polar-crossing model vs SGP4 (r = {rho:.2f})")
    fig.tight_layout(); fig.savefig(OUT / "superseded" / "pilot_fig6_crossing_model.png", dpi=150)


if __name__ == "__main__":
    main()
