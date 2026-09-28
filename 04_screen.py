"""Small close-approach screen: Qianfan vs Qianfan, vs CZ-6A (2024-140) fragments, vs background.

At each screening epoch T0 every object uses its element set nearest to T0 (within --max-age-h),
is propagated with SGP4 over --span-h hours on a --step-s grid, and candidate pairs come from a
KD-tree search with radius thr + v_max*step/2 (so no encounter closer than thr can fall between
samples). Every local minimum of a candidate window is refined to the time of closest approach.

Outputs d_min, v_rel, latitude and altitude at TCA, the element-set age of both objects at TCA,
and the transition state of the Qianfan satellite(s). No collision probability is computed.

Usage:
    py 04_screen.py                              # Space-Track history, one epoch every 30 days
    py 04_screen.py --every-days 14 --span-h 24
    py 04_screen.py --snapshot data/qf_celestrak.json   # quick test on one CelesTrak snapshot
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.stats import chi2
from sgp4.api import SatrecArray

from common import DATA, OUT, latitude_deg, load_gp, load_states, RE, satrec, to_jd

VMAX = 16.0      # km/s, upper bound on LEO relative speed
CHUNK = 360      # time steps propagated per batch
THRESHOLDS = (5, 10, 20)
TRANSIT = ("drift", "ascent")


def nearest(df: pd.DataFrame, t0: pd.Timestamp, max_age_h: float) -> pd.DataFrame:
    dt = (df["EPOCH"] - t0).abs()
    d = df[dt <= pd.Timedelta(hours=max_age_h)].assign(_dt=dt)
    if d.empty:
        return d
    return d.loc[d.groupby("NORAD_CAT_ID")["_dt"].idxmin()].drop(columns="_dt").reset_index(drop=True)


def park_invalid(e: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Move objects SGP4 could not propagate far away, each to its own spot."""
    bad = (e != 0) | ~np.isfinite(r).all(-1)
    if bad.any():
        far = np.zeros_like(r)
        far[..., 0] = 1e8 * (1 + np.arange(r.shape[0]))[:, None]
        r = np.where(bad[..., None], far, r)
    return r


def screen(prim: pd.DataFrame, sec: pd.DataFrame | None, t0: pd.Timestamp,
           span_h: float, step_s: float, thr: float) -> pd.DataFrame:
    """Encounters with d_min <= thr. sec=None screens the primaries against each other."""
    ps = [satrec(r) for _, r in prim.iterrows()]
    ss = ps if sec is None else [satrec(r) for _, r in sec.iterrows()]
    PA = SatrecArray(ps)
    SA = PA if sec is None else SatrecArray(ss)
    n = int(span_h * 3600 / step_s) + 1
    jd0, fr0 = to_jd(t0)
    jd = np.full(n, jd0[0])
    fr = fr0[0] + np.arange(n) * step_s / 86400
    rc = thr + VMAX * step_s / 2

    hits = []
    for c0 in range(0, n, CHUNK):
        sl = slice(c0, c0 + CHUNK)
        eP, rP, _ = PA.sgp4(jd[sl], fr[sl])
        rP = park_invalid(eP, rP)
        if sec is None:
            rS = rP
        else:
            eS, rS, _ = SA.sgp4(jd[sl], fr[sl])
            rS = park_invalid(eS, rS)
        for m in range(rP.shape[1]):
            tp = cKDTree(rP[:, m])
            ts = tp if sec is None else cKDTree(rS[:, m])
            sp = tp.sparse_distance_matrix(ts, rc, output_type="ndarray")
            if sec is None:
                sp = sp[sp["i"] < sp["j"]]
            if len(sp):
                hits.append(np.column_stack([sp["i"], sp["j"], np.full(len(sp), c0 + m), sp["v"]]))
    if not hits:
        return pd.DataFrame()

    H = pd.DataFrame(np.vstack(hits), columns=["i", "j", "k", "d"]).astype({"i": int, "j": int, "k": int})
    H = H.sort_values(["i", "j", "k"]).reset_index(drop=True)
    new = (H["i"].diff() != 0) | (H["j"].diff() != 0) | (H["k"].diff() != 1)
    H["w"] = new.cumsum()

    def dist(s, a, b):
        _, ra, _ = a.sgp4(jd0[0], fr0[0] + s / 86400)
        _, rb, _ = b.sgp4(jd0[0], fr0[0] + s / 86400)
        return np.linalg.norm(np.subtract(ra, rb))

    events = []
    for _, w in H.groupby("w"):
        d = w["d"].to_numpy()
        prev = np.concatenate([[np.inf], d[:-1]])
        nxt = np.concatenate([d[1:], [np.inf]])
        i, j = int(w["i"].iloc[0]), int(w["j"].iloc[0])
        a, b = ps[i], ss[j]
        for idx in np.flatnonzero((d <= prev) & (d < nxt)):
            s0 = w["k"].iloc[idx] * step_s
            opt = minimize_scalar(dist, bounds=(max(s0 - step_s, 0), s0 + step_s), args=(a, b),
                                  method="bounded", options={"xatol": 1e-3})
            if opt.fun > thr:
                continue
            _, ra, va = a.sgp4(jd0[0], fr0[0] + opt.x / 86400)
            _, _, vb = b.sgp4(jd0[0], fr0[0] + opt.x / 86400)
            ra = np.array(ra)
            events.append(dict(i=i, j=j, tca=t0 + pd.Timedelta(seconds=float(opt.x)), d_min_km=opt.fun,
                               v_rel_kms=float(np.linalg.norm(np.subtract(va, vb))),
                               lat_deg=float(latitude_deg(ra)), alt_km=float(np.linalg.norm(ra) - RE),
                               window_min=len(w) * step_s / 60))
    return pd.DataFrame(events)


def annotate(ev: pd.DataFrame, prim: pd.DataFrame, sec: pd.DataFrame, kind: str) -> pd.DataFrame:
    P = prim.add_suffix("_1").reset_index(drop=True)
    S = sec.add_suffix("_2").reset_index(drop=True)
    ev = ev.join(P, on="i").join(S, on="j")
    ev["age_1_h"] = (ev["tca"] - ev["EPOCH_1"]).abs().dt.total_seconds() / 3600
    ev["age_2_h"] = (ev["tca"] - ev["EPOCH_2"]).abs().dt.total_seconds() / 3600
    if kind == "QQ":
        ev["pair_type"] = np.where(ev["LAUNCH_1"] == ev["LAUNCH_2"], "QQ_same_launch", "QQ_diff_launch")
    else:
        ev["pair_type"] = kind
    keep = ["tca", "pair_type", "d_min_km", "v_rel_kms", "lat_deg", "alt_km", "window_min",
            "age_1_h", "age_2_h", "NORAD_CAT_ID_1", "OBJECT_ID_1", "STATE_1",
            "NORAD_CAT_ID_2", "OBJECT_ID_2", "STATE_2", "OBJECT_TYPE_2"]
    return ev[[c for c in keep if c in ev]]


def poisson_ci(k: int) -> tuple[float, float]:
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return lo, chi2.ppf(0.975, 2 * k + 2) / 2


def pair_table(ev: pd.DataFrame, span_h: float, max_age_tca: float) -> pd.DataFrame:
    """Collapse repeated passes into one row per pair and screening window.

    Two satellites in different near-polar planes that cross the polar intersection almost in step
    meet on every half-orbit, so passes are strongly dependent. The pair is the counting unit.
    """
    ok = ev[(ev["age_1_h"] <= max_age_tca) & (ev["age_2_h"] <= max_age_tca)]
    keys = ["t0", "pair_type", "NORAD_CAT_ID_1", "NORAD_CAT_ID_2"]
    pairs = ok.groupby(keys).agg(
        passes=("d_min_km", "size"), d_min_km=("d_min_km", "min"), d_med_km=("d_min_km", "median"),
        v_rel_kms=("v_rel_kms", "median"), abs_lat_deg=("lat_deg", lambda x: x.abs().median()),
        STATE_1=("STATE_1", "first"), STATE_2=("STATE_2", "first"),
        OBJECT_ID_1=("OBJECT_ID_1", "first"), OBJECT_ID_2=("OBJECT_ID_2", "first")).reset_index()
    polar_crossings = 2 * span_h * 60 / 107          # ~107 min period at ~1070 km
    pairs["persistent"] = pairs["passes"] >= 0.5 * polar_crossings
    return pairs


def summarise(pairs: pd.DataFrame, exposure: dict) -> pd.DataFrame:
    """Pairs per 100 satellite-days by state (a QQ pair counts once for each Qianfan member).

    exposure is keyed by (catalogue, state): a catalogue only accrues satellite-days in the windows
    where it was actually screened (the background snapshot only covers recent windows).
    """
    parts = [pairs.assign(STATE=pairs["STATE_1"])]
    qq = pairs[pairs["pair_type"].str.startswith("QQ")]
    parts.append(qq.assign(STATE=qq["STATE_2"]))
    sp = pd.concat(parts)
    rows = []
    for thr in THRESHOLDS:
        for (pt, st), g in sp[sp["d_min_km"] <= thr].groupby(["pair_type", "STATE"]):
            k = len(g)
            sd = exposure.get(("QQ" if pt.startswith("QQ") else pt, st), np.nan)
            lo, hi = poisson_ci(k)
            rows.append(dict(threshold_km=thr, pair_type=pt, state=st, pairs=k,
                             persistent=int(g["persistent"].sum()), sat_days=sd,
                             per_100sd=100 * k / sd, ci95_lo=100 * lo / sd, ci95_hi=100 * hi / sd))
    return pd.DataFrame(rows).round(3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", help="screen one Qianfan snapshot file (e.g. from CelesTrak) instead")
    ap.add_argument("--every-days", type=float, default=30)
    ap.add_argument("--start", help="first screening epoch (UTC date); default: start of history")
    ap.add_argument("--end", help="last screening epoch (UTC date); default: end of history")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _daily")
    ap.add_argument("--kinds", default="QQ,QB_CZ6A,QB_other", help="comma list of QQ, QB_CZ6A, QB_other")
    ap.add_argument("--span-h", type=float, default=24)
    ap.add_argument("--step-s", type=float, default=10)
    ap.add_argument("--thr", type=float, default=max(THRESHOLDS))
    ap.add_argument("--max-age-h", type=float, default=24, help="element-set age allowed at T0")
    ap.add_argument("--max-age-tca", type=float, default=36, help="element-set age allowed at TCA for rates")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    if args.snapshot:
        qf = load_gp([args.snapshot])
        epochs = [qf["EPOCH"].median().floor("h")]
        args.max_age_h = max(args.max_age_h, 72)
    else:
        qf = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
        start = pd.Timestamp(args.start, tz="UTC") if args.start else qf["EPOCH"].min().ceil("D") + pd.Timedelta(days=1)
        end = pd.Timestamp(args.end, tz="UTC") if args.end else qf["EPOCH"].max().floor("D") - pd.Timedelta(days=1)
        epochs = list(pd.date_range(start, end, freq=f"{args.every_days}D"))
    states = load_states()
    qf["STATE"] = "unknown"
    if states is not None:
        qf = qf.drop(columns="STATE").merge(states[["NORAD_CAT_ID", "EPOCH", "STATE"]],
                                            on=["NORAD_CAT_ID", "EPOCH"], how="left")
        qf["STATE"] = qf["STATE"].fillna("unknown")

    cz_files = sorted((DATA / "gp_history_cz6a").glob("*.json.gz"))
    cz = load_gp(cz_files) if cz_files else None
    bg_file = DATA / "background_snapshot.json.gz"
    bg = load_gp([bg_file]) if bg_file.exists() else None
    if bg is not None:
        bg = bg[~bg["NORAD_CAT_ID"].isin(qf["NORAD_CAT_ID"]) & ~bg["OBJECT_ID"].str.startswith("2024-140")]

    all_ev, exposure = [], {}
    for t0 in epochs:
        prim = nearest(qf, t0, args.max_age_h)
        if len(prim) < 2:
            continue
        lo =prim["PERIGEE_KM"].min() - args.thr - 5
        hi = prim["APOGEE_KM"].max() + args.thr + 5
        jobs = [("QQ", None)] if "QQ" in args.kinds.split(",") else []
        for kind, cat in (("QB_CZ6A", cz), ("QB_other", bg)):
            if cat is None or kind not in args.kinds.split(","):
                continue
            sec = nearest(cat, t0, args.max_age_h)
            sec = sec[(sec["APOGEE_KM"] >= lo) & (sec["PERIGEE_KM"] <= hi)]
            if len(sec):
                jobs.append((kind, sec))
        for kind, sec in jobs:
            for s, n in prim["STATE"].value_counts().items():
                exposure[(kind, s)] = exposure.get((kind, s), 0) + n * args.span_h / 24
            ev = screen(prim, sec, t0, args.span_h, args.step_s, args.thr)
            n_sec = len(prim) if sec is None else len(sec)
            print(f"{t0:%Y-%m-%d %H:%M}  {kind:9s} {len(prim):4d} x {n_sec:5d} objects -> {len(ev)} encounters <= {args.thr} km")
            if len(ev):
                all_ev.append(annotate(ev, prim, prim if sec is None else sec, kind).assign(t0=t0))

    if not all_ev:
        print("no encounters found")
        return
    ev = pd.concat(all_ev, ignore_index=True)
    ev.to_csv(OUT / f"encounters{args.tag}.csv", index=False)
    pairs = pair_table(ev, args.span_h, args.max_age_tca)
    pairs.to_csv(OUT / f"encounter_pairs{args.tag}.csv", index=False)
    summ = summarise(pairs, exposure)
    summ.to_csv(OUT / f"encounter_rates{args.tag}.csv", index=False)

    print(f"\n{len(ev)} passes, {len(pairs)} distinct pairs in {len(epochs)} windows of {args.span_h:g} h "
          f"-> {OUT / f'encounter_pairs{args.tag}.csv'}")
    print("satellite-days screened (catalogue, state): " + ", ".join(f"{k}/{s} {v:.0f}" for (k, s), v in exposure.items()))
    print("\npasses vs distinct pairs (all element-set ages):")
    tab = {}
    for t in THRESHOLDS:
        e = ev[ev["d_min_km"] <= t]
        tab[f"passes <= {t} km"] = e["pair_type"].value_counts()
        tab[f"pairs <= {t} km"] = e.groupby("pair_type")[["NORAD_CAT_ID_1", "NORAD_CAT_ID_2"]].apply(
            lambda g: len(g.drop_duplicates()))
    print(pd.DataFrame(tab).fillna(0).astype(int).to_string())
    print(f"\npairs per 100 satellite-days (element-set age at TCA <= {args.max_age_tca:g} h):\n"
          f"{summ.to_string(index=False)}")

    print("\n==== GO / NO-GO ====")
    t10 = summ[(summ["threshold_km"] == 10) & summ["state"].isin(TRANSIT) & (summ["pair_type"] != "QQ_same_launch")]
    k = int(t10["pairs"].sum())
    print(f"[{'PASS' if k >= 30 else 'FAIL'}] distinct 10 km pairs involving a transit-state satellite, excluding "
          f"same-launch pairs: {k} (rule of thumb >= 30; rerun with smaller --every-days if close)")
    n_pers = int(pairs.loc[(pairs["d_min_km"] <= 10) & pairs["pair_type"].str.startswith("QQ"), "persistent"].sum())
    print(f"[info] persistent Qianfan-Qianfan pairs at 10 km (meeting on >= half of polar crossings): {n_pers}")
    if states is None:
        print("(states unknown: run 02_states.py on Space-Track history to get the transit/operational split)")


if __name__ == "__main__":
    main()
