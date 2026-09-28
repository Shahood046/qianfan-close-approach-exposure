"""Infer the as-deployed plane / phasing structure of the operational shell and compare the
closest approaches it produces with those of an idealised version of the same pattern.

1. Take operational satellites at one epoch, propagate with SGP4 to that epoch, and compute
   RAAN and argument of latitude u from the osculating state.
2. Cluster RAAN into planes (gap > --gap deg). Estimate the in-plane slot spacing s from the
   median gap in u, and each plane's phase (circular mean of u mod s).
3. Reference pattern: same planes (mean RAAN), same slot phases, every satellite snapped to its
   nearest slot, one common *circular* radius and inclination. Relative geometry within a shell
   is periodic, so one orbital period of two-body motion gives each pair's minimum distance.
4. Observed: the same pairs screened with SGP4 over one period.

The reference has no radial separation at the polar crossings, so the comparison isolates what
the real orbits add on top of along-track phasing: slot errors, and radial separation from each
plane's eccentricity vector (argument of perigee differs between planes). It is not the operator's
design, which is not public. Walker phasing studies aim for minimum separations of tens of km.

Usage:  py 05_planes.py [--date 2026-09-01] [--snapshot data/qf_celestrak.json]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from common import DATA, MU, OUT, load_gp, load_states, satrec, to_jd

STEP_S = 5.0


def elements_at(rows: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    jd, fr = to_jd(t)
    rs, vs, um = [], [], []
    for _, row in rows.iterrows():
        sat = satrec(row)
        _, r, v = sat.sgp4(jd[0], fr[0])
        rs.append(r), vs.append(v)
        um.append(np.degrees(sat.om + sat.mm) % 360)     # SGP4 mean argument of latitude after propagation
    r, v = np.array(rs), np.array(vs)
    h = np.cross(r, v)
    raan = np.degrees(np.arctan2(h[:, 0], -h[:, 1])) % 360
    inc = np.degrees(np.arccos(h[:, 2] / np.linalg.norm(h, axis=1)))
    node = np.stack([np.cos(np.radians(raan)), np.sin(np.radians(raan)), np.zeros(len(r))], 1)
    y = np.cross(h / np.linalg.norm(h, axis=1, keepdims=True), node)
    u = np.degrees(np.arctan2((r * y).sum(1), (r * node).sum(1))) % 360
    out = rows[["NORAD_CAT_ID", "OBJECT_ID", "LAUNCH"]].copy()
    out["raan"], out["inc"], out["u"], out["rad"] = raan, inc, u, np.linalg.norm(r, axis=1)
    out["um"] = um
    return out.reset_index(drop=True)


def cluster_planes(raan: np.ndarray, gap: float) -> np.ndarray:
    order = np.argsort(raan)
    r = raan[order]
    brk = np.r_[False, np.diff(r) > gap]
    lab = np.cumsum(brk)
    if r[0] + 360 - r[-1] <= gap:      # wrap-around: first and last clusters are one plane
        lab[lab == lab.max()] = 0
    out = np.empty_like(lab)
    out[order] = lab
    return out


def circmean(deg: np.ndarray, period: float = 360) -> float:
    a = np.exp(2j * np.pi * np.asarray(deg) / period)
    return (np.angle(a.mean()) * period / (2 * np.pi)) % period


def positions(raan, inc, u0, rad, t):
    """Circular two-body positions, shape (n, len(t), 3)."""
    n = np.sqrt(MU / rad**3)
    u = np.radians(u0)[:, None] + n[:, None] * t[None, :]
    O, i = np.radians(raan)[:, None], np.radians(inc)[:, None]
    return np.stack([np.cos(O) * np.cos(u) - np.sin(O) * np.sin(u) * np.cos(i),
                     np.sin(O) * np.cos(u) + np.cos(O) * np.sin(u) * np.cos(i),
                     np.sin(u) * np.sin(i)], -1) * rad[:, None, None]


def min_distances(pos_fn, n_obj: int, period: float, near_km: float = 100) -> np.ndarray:
    """Minimum distance over one period for every pair (i < j)."""
    t = np.arange(0, period + STEP_S, STEP_S)
    p = pos_fn(t)
    iu, ju = np.triu_indices(n_obj, 1)
    dmin = np.full(len(iu), np.inf)
    kmin = np.zeros(len(iu), int)
    for c in range(0, len(iu), 2000):
        sl = slice(c, c + 2000)
        d = np.linalg.norm(p[iu[sl]] - p[ju[sl]], axis=-1)
        kmin[sl], dmin[sl] = d.argmin(1), d.min(1)
    for m in np.flatnonzero(dmin < near_km):       # refine the close ones between samples
        f = lambda s: np.linalg.norm(np.subtract(*pos_fn(np.array([s]))[[iu[m], ju[m]], 0]))
        t0 = t[kmin[m]]
        dmin[m] = min(dmin[m], minimize_scalar(f, bounds=(t0 - STEP_S, t0 + STEP_S), method="bounded").fun)
    return dmin


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="epoch (UTC); default: one day before the last element set")
    ap.add_argument("--snapshot")
    ap.add_argument("--X", type=float, default=15, help="operational band half-width if no states file")
    ap.add_argument("--gap", type=float, default=2.0, help="RAAN gap separating planes, deg")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    files = [args.snapshot] if args.snapshot else sorted((DATA / "gp_history_qianfan").glob("*.json.gz"))
    df = load_gp(files)
    t = pd.Timestamp(args.date, tz="UTC") if args.date else df["EPOCH"].max().floor("h") - pd.Timedelta(days=1 if not args.snapshot else 0)
    near = df.assign(_dt=(df["EPOCH"] - t).abs())
    near = near[near["_dt"] <= pd.Timedelta(hours=48)]
    near = near.loc[near.groupby("NORAD_CAT_ID")["_dt"].idxmin()]
    states = load_states()
    if states is not None and not args.snapshot:
        near = near.merge(states[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"])
        op = near[near["STATE"] == "operational"]
    else:
        mode = near["ALT_KM"].round(-1).mode().iloc[0]
        op = near[(near["ALT_KM"] - near.loc[(near["ALT_KM"] - mode).abs() < 20, "ALT_KM"].median()).abs() <= args.X]
    el = elements_at(op, t)
    el["plane"] = cluster_planes(el["raan"].to_numpy(), args.gap)

    gaps = [np.median(np.diff(np.sort(g["u"].to_numpy()))) for _, g in el.groupby("plane") if len(g) > 2]
    s = 360 / round(360 / min(gaps))                 # finest slot spacing seen in any plane
    planes = el.groupby("plane").agg(n=("u", "size"), raan=("raan", circmean),
                                     raan_spread=("raan", lambda x: np.ptp(np.unwrap(np.radians(x)) * 180 / np.pi)),
                                     inc=("inc", "mean"), launches=("LAUNCH", lambda x: ",".join(sorted(set(x)))))
    planes["phase"] = el.groupby("plane")["u"].apply(lambda u: circmean(u % s, s))
    planes = planes.sort_values("raan")
    planes["d_raan"] = planes["raan"].diff()
    planes["d_phase"] = (planes["phase"].diff()) % s
    # slot residual: how far each satellite sits from its plane's slot grid
    el["slot_err_deg"] = (el["u"] - el["plane"].map(planes["phase"]) + s / 2) % s - s / 2

    print(f"epoch {t:%Y-%m-%d %H:%M} UTC, {len(el)} operational satellites, {len(planes)} planes, "
          f"slot spacing {s:.2f} deg")
    print(planes.round(3).to_string())
    print(f"\nslot error (deg): median |err| {el['slot_err_deg'].abs().median():.3f}, "
          f"95th pct {el['slot_err_deg'].abs().quantile(0.95):.3f}; "
          f"radius spread (km): {np.ptp(el['rad']):.1f}; inclination sd {el['inc'].std():.4f} deg")

    # ---- ideal vs observed minimum distances -----------------------------------------
    rad0, inc0 = el["rad"].median(), el["inc"].mean()
    period = 2 * np.pi * np.sqrt(rad0**3 / MU)
    ideal_u = el["u"] - el["slot_err_deg"]
    ideal_raan = el["plane"].map(planes["raan"]).to_numpy()
    n = len(el)
    ideal = min_distances(lambda tt: positions(ideal_raan, np.full(n, inc0), ideal_u.to_numpy(),
                                               np.full(n, rad0), tt), n, period)
    sats = [satrec(r) for _, r in op.iterrows()]
    jd0, fr0 = to_jd(t)
    def sgp4_pos(tt):
        return np.stack([np.array(sa.sgp4_array(np.full(len(tt), jd0[0]), fr0[0] + tt / 86400)[1]) for sa in sats])
    observed = min_distances(sgp4_pos, n, period)

    iu, ju = np.triu_indices(n, 1)
    pl = el["plane"].to_numpy()
    res = pd.DataFrame({"id_1": el["OBJECT_ID"].to_numpy()[iu], "id_2": el["OBJECT_ID"].to_numpy()[ju],
                        "plane_1": pl[iu], "plane_2": pl[ju], "ideal_dmin_km": ideal, "observed_dmin_km": observed})
    res = res[res["plane_1"] != res["plane_2"]]
    res.to_csv(OUT / "ideal_vs_observed_pairs.csv", index=False)
    print("\ninter-plane pairs within threshold over one orbit:")
    print(pd.DataFrame({"circular reference": [(res["ideal_dmin_km"] <= k).sum() for k in (5, 10, 20, 50)],
                        "as deployed (SGP4)": [(res["observed_dmin_km"] <= k).sum() for k in (5, 10, 20, 50)]},
                       index=["<= 5 km", "<= 10 km", "<= 20 km", "<= 50 km"]).to_string())
    print(f"smallest inter-plane separation: circular reference {res['ideal_dmin_km'].min():.2f} km, "
          f"as deployed {res['observed_dmin_km'].min():.2f} km  -> {OUT / 'ideal_vs_observed_pairs.csv'}")
    ecc = op.set_index("NORAD_CAT_ID")[["ECCENTRICITY", "ARG_OF_PERICENTER"]]
    per_plane = el.join(ecc, on="NORAD_CAT_ID").groupby("plane").agg(
        ecc=("ECCENTRICITY", "median"), argp=("ARG_OF_PERICENTER", circmean))
    print("\nper-plane mean eccentricity vector (radial offset at the pole ~ a*e*sin(argp)):")
    print(per_plane.join(planes["raan"]).sort_values("raan").round(5).to_string())


if __name__ == "__main__":
    main()
