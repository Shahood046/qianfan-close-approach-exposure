"""Phase-error dynamics for manuscript Section 6, written to PHASE_CLAIMS.md (+ out/phase_turning_points.csv, out/fig8_picks.csv for Figure 8, drawn by 27_fig8_trajectories.py).

Quantities (all in mean argument of latitude, degrees):
    absolute phase error   |eps|, eps = u_b - (nearest slot of the satellite's plane grid), wrapped to +-5 deg
    signed phase error     eps
    change in phase error  eps(t + L) - eps(t) for the same satellite (unwrapped with period 10 deg)
    relative pair drift    r cos(gamma/2) |n_a - n_b| in km/day (the pair's along-track drift at the crossing, 11_per_angle)
    pair phase deviation   eps_a - eps_b = Delta phi_observed - Delta phi_nominal at the crossing; as along-track distance
                           |eps_a - eps_b| * r * cos(gamma/2) (km)

A. Daily table of every operational satellite's eps at each window centre (same element-set selection and grid estimator
   as 11_per_angle.py), cached in data/slot_errors_daily.pkl.gz.
B. 6.1 distribution by month and by time since arrival (sparse 2024-185 plane excluded).
C. 6.2 persistence: satellite-matched correlation of signed eps at lags 1, 7, 30, 60, 90, 180 d.
D. 6.3 signed trajectories: rates, reversals, excursions, slot migrations; figure of representative satellites.
E. 6.4 bridge to exposure: pair phase deviation vs nominal margin, events vs non-events matched by nominal-margin bin.

Usage:  py 19_phase_dynamics.py [--rebuild]
"""
from __future__ import annotations

import argparse
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DATA, OUT, RE, ROOT, load_gp, load_states, satrec, to_jd

_spec = importlib.util.spec_from_file_location("rb", ROOT / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
_spec2 = importlib.util.spec_from_file_location("geo", ROOT / "17_geometry.py")
geo = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(geo)

S = 10.0
R_KM = RE + 1068.5
CACHE = DATA / "slot_errors_daily.pkl.gz"
LAGS = [1, 7, 30, 60, 90, 180]
SPARSE_LAUNCH = "2024-185"


def build(windows) -> pd.DataFrame:
    """Daily satellite-level phase errors in two phase variables:
    eps_ub  from u_b = u - 2e sin(u - w) (equation of centre with the element-set e, as in the crossing model)
    eps_um  from SGP4's own mean argument of latitude (om + mm after propagation), which also removes the long-period
            (J3) eccentricity term that u_b retains: u_b - u_m = -0.115 deg cos u, identical for all satellites.
    The term cancels between two satellites crossing near the pole (both near u = 90 or 270 deg) but not between
    satellites at different positions in a plane, so satellite-level statistics use eps_um."""
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    frames = []
    for n, t0 in enumerate(windows):
        t = pd.Timestamp(t0) + pd.Timedelta(hours=12)
        near = df.assign(_dt=(df["EPOCH"] - t).abs())
        near = near[near["_dt"] <= pd.Timedelta(hours=48)]
        near = near.loc[near.groupby("NORAD_CAT_ID")["_dt"].idxmin()].merge(
            states[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"])
        op = near[near["STATE"] == "operational"].reset_index(drop=True)
        try:
            el = rb.get_el(df, states, t)
        except ValueError:
            continue
        if not (op["NORAD_CAT_ID"].to_numpy() == el["NORAD_CAT_ID"].to_numpy()).all():
            raise RuntimeError(f"element-set order mismatch at {t}")
        jd, fr = to_jd(t)
        um = []
        for _, r in op.iterrows():
            s = satrec(r)
            s.sgp4(jd[0], fr[0])
            um.append(np.degrees(s.om + s.mm) % 360)
        e_ub = rb.slots(el, rb.ESTIMATORS["circular mean"], col="ub")
        e_um = rb.slots(el, rb.ESTIMATORS["circular mean"], col="um")
        frames.append(el[["NORAD_CAT_ID", "OBJECT_ID", "LAUNCH", "plane", "u"]].assign(
            t=t, um=um, eps_ub=e_ub["slot_err_deg"].to_numpy(), eps_um=e_um["slot_err_deg"].to_numpy()))
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(windows)} windows", flush=True)
    E = pd.concat(frames, ignore_index=True)
    E.to_pickle(CACHE)
    return E


def unwrap_group(g: pd.DataFrame) -> pd.Series:
    """eps unwrapped in time with period S (a satellite that passes +-5 deg keeps counting instead of jumping)."""
    x = np.radians(g["slot_err_deg"].to_numpy() * 360 / S)
    return pd.Series(np.degrees(np.unwrap(x)) * S / 360, index=g.index)


def md(df):
    return geo.md_table(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    P = pd.read_pickle(DATA / "pair_windows_daily.pkl.gz")
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    E = pd.read_pickle(CACHE) if CACHE.exists() and not args.rebuild else build(sorted(P["t0"].unique()))
    sats = pd.read_csv(OUT / "satellite_summary.csv")
    sats["arrival"] = pd.to_datetime(sats["arrival"], utc=True, format="ISO8601")
    E = E.merge(sats[["NORAD_CAT_ID", "arrival"]], on="NORAD_CAT_ID", how="left")
    E["age_d"] = (E["t"] - E["arrival"]).dt.total_seconds() / 86400
    E = E[E["LAUNCH"] != SPARSE_LAUNCH].sort_values(["NORAD_CAT_ID", "t"]).reset_index(drop=True)
    E["slot_err_deg"] = E["eps_um"]                      # satellite-level phase error: SGP4 mean argument of latitude
    E["abs"] = E["slot_err_deg"].abs()
    E["eps_u"] = E.groupby("NORAD_CAT_ID", group_keys=False).apply(unwrap_group)
    E["day"] = (E["t"] - E["t"].min()).dt.days
    lines = ["# Phase-error dynamics for Section 6 (auto-generated by 19_phase_dynamics.py)", "",
             f"Daily table: {len(E):,} satellite-days, {E['NORAD_CAT_ID'].nunique()} satellites, {E['t'].nunique()} windows "
             f"({E['t'].min():%Y-%m-%d} to {E['t'].max():%Y-%m-%d}); sparse plane (launch {SPARSE_LAUNCH}) excluded", ""]
    add = lines.append

    add("## Phase variable check")
    for col in ("eps_ub", "eps_um"):
        Wc = E.pivot_table(index="day", columns="NORAD_CAT_ID", values=col)
        ch = (Wc.diff().abs().stack())
        ch = ch[ch < 5]
        add(f"- {col}: median |eps| {E[col].abs().median():.3f} deg; median 1-day |change| (same satellite) {ch.median():.3f} deg")
    dd_ = ((E["eps_ub"] - E["eps_um"] + 5) % 10) - 5
    X = np.column_stack([np.ones(len(E)), np.cos(np.radians(E["u"])), np.sin(np.radians(E["u"]))])
    cf, *_ = np.linalg.lstsq(X, dd_ - dd_.groupby([E["t"], E["plane"]]).transform("mean"), rcond=None)
    add(f"- eps_ub - eps_um (plane-mean removed) = {cf[1]:+.3f} cos u {cf[2]:+.3f} sin u deg (all satellite-days); "
        "the long-period eccentricity term retained by the equation-of-centre phase")
    add("")

    # ---------------- B. distribution over time and by time since arrival
    add("## 6.1 Absolute phase error by month (all operational satellites of populated planes)")
    E["month"] = E["t"].dt.tz_convert(None).dt.to_period("M").astype(str)
    M = E.groupby("month")["abs"].agg(n="size", median="median", p90=lambda x: x.quantile(.9), p95=lambda x: x.quantile(.95),
                                     within1=lambda x: (x <= 1).mean())
    M["signed_median"] = E.groupby("month")["slot_err_deg"].median()
    M["signed_p5"] = E.groupby("month")["slot_err_deg"].quantile(.05)
    M["signed_p95"] = E.groupby("month")["slot_err_deg"].quantile(.95)
    M["satellites"] = E.groupby("month")["NORAD_CAT_ID"].nunique()
    add("")
    add(md(M.reset_index().round(3)))
    add("")
    bins = [0, 30, 90, 180, 10000]
    E["age_bin"] = pd.cut(E["age_d"], bins, right=False, labels=["0-30 d", "30-90 d", "90-180 d", ">= 180 d"])
    A = E.groupby("age_bin", observed=True)["abs"].agg(n="size", median="median", p90=lambda x: x.quantile(.9),
                                                       p99=lambda x: x.quantile(.99), within1=lambda x: (x <= 1).mean())
    add("By time since arrival (satellite-days):")
    add("")
    add(md(A.reset_index().round(3)))
    add("")
    ok = E["abs"] <= 1
    first_on = E[ok].groupby("NORAD_CAT_ID")["age_d"].min()
    add(f"- Days from arrival to first day within 1 deg of the grid: median {first_on.median():.0f}, P90 {first_on.quantile(.9):.0f} "
        f"(n = {len(first_on)} satellites; negative = already on grid at arrival)")
    add("")

    # ---------------- C. persistence
    add("## 6.2 Persistence of the signed phase error (same satellite, lag L)")
    add("")
    W = E.pivot_table(index="day", columns="NORAD_CAT_ID", values="slot_err_deg")
    Wabs = W.abs()
    Wage = E.pivot_table(index="day", columns="NORAD_CAT_ID", values="age_d")
    rows = []
    for L in LAGS:
        a, b = W.iloc[:-L].to_numpy().ravel(), W.iloc[L:].to_numpy().ravel()
        ag = Wage.iloc[:-L].to_numpy().ravel()
        on = (np.abs(a) <= 1) & (np.abs(b) <= 1)
        m = np.isfinite(a) & np.isfinite(b)
        for name, sel in (("all", m), ("on grid at both dates", m & on), ("on grid, >= 90 d after arrival", m & on & (ag >= 90))):
            if sel.sum() > 50:
                rows.append(dict(lag_d=L, subset=name, n=int(sel.sum()), r=round(float(np.corrcoef(a[sel], b[sel])[0, 1]), 3),
                                 median_abs_change=round(float(np.median(np.abs(a[sel] - b[sel]))), 3)))
    add(md(pd.DataFrame(rows)))
    add("")

    # ---------------- D. signed trajectories
    add("## 6.3 Signed phase evolution")
    E["rate"] = E.groupby("NORAD_CAT_ID")["eps_u"].diff() / E.groupby("NORAD_CAT_ID")["day"].diff()
    steady = E[(E["age_d"] >= 90) & (E["abs"] <= 1)]
    add(f"- Daily rate of change |d eps/dt| (satellites >= 90 d after arrival, on grid): median {steady['rate'].abs().median():.3f} deg/day, "
        f"P90 {steady['rate'].abs().quantile(.9):.3f} (= {steady['rate'].abs().median() * np.pi / 180 * R_KM:.1f} / "
        f"{steady['rate'].abs().quantile(.9) * np.pi / 180 * R_KM:.1f} km/day along-track, one satellite)")
    rev_rows, mig = [], []
    for sid, g in E.groupby("NORAD_CAT_ID"):
        g = g[g["age_d"] >= 90]
        if len(g) < 60:
            continue
        x = g.set_index("day")["eps_u"].reindex(range(g["day"].min(), g["day"].max() + 1)).interpolate(limit=3)
        xs = x.rolling(7, center=True, min_periods=4).mean()
        sl = np.sign(xs.diff())
        # turning points: slope sign changes that are separated by an excursion of at least 0.05 deg
        tp, last_v, last_s = [], None, 0
        for d, s, v in zip(xs.index, sl.to_numpy(), xs.to_numpy()):
            if not np.isfinite(v) or s == 0 or not np.isfinite(s):
                continue
            if last_s and s != last_s and (last_v is None or abs(v - last_v) >= 0.05):
                tp.append((d, v))
                last_v = v
            if last_v is None:
                last_v = v
            last_s = s
        span = x.index.max() - x.index.min()
        if len(tp) >= 2:
            gaps = np.diff([d for d, _ in tp])
            amp = np.abs(np.diff([v for _, v in tp]))
            rev_rows.append(dict(sid=sid, days=span, reversals=len(tp), gap_med=np.median(gaps), amp_med=np.median(amp)))
        else:
            rev_rows.append(dict(sid=sid, days=span, reversals=len(tp), gap_med=np.nan, amp_med=np.nan))
        slot_idx = np.round((g["eps_u"] - g["slot_err_deg"]) / S)      # changes when the nearest slot changes
        mig.append(dict(sid=sid, migrations=int((np.abs(np.diff(slot_idx.to_numpy())) > 0).sum()),
                        range_deg=float(g["eps_u"].max() - g["eps_u"].min())))
    RV, MG = pd.DataFrame(rev_rows), pd.DataFrame(mig)
    RV.to_csv(OUT / "phase_turning_points.csv", index=False)     # per-satellite turning-point statistics (Figure 8 panel b)
    MG.to_csv(OUT / "phase_migrations.csv", index=False)

    # Trajectory classes over consecutive 30-day segments of each satellite's post-arrival record (eps unwrapped):
    #   migrating    the nearest slot changes within the segment
    #   approaching  |eps| falls by >= 0.3 deg from the first to the last week and ends within 1 deg, no turning point
    #   reversing    >= 1 turning point of the 7-day smoothed eps with >= 0.05 deg between turns
    #   stationary   range of the smoothed eps < 0.1 deg (none of the above)
    #   drifting     monotonic change without approach (none of the above)
    seg_rows = []
    for sid, g in E[E["age_d"] >= 0].groupby("NORAD_CAT_ID"):
        x = g.set_index("day")[["eps_u", "slot_err_deg"]]
        x = x.reindex(range(x.index.min(), x.index.max() + 1)).interpolate(limit=3)
        for s0 in range(x.index.min(), x.index.max() - 29, 30):
            w = x.loc[s0:s0 + 29].dropna()
            if len(w) < 25:
                continue
            sm = w["eps_u"].rolling(7, center=True, min_periods=4).mean().dropna()
            slope = np.sign(np.diff(sm.to_numpy()))
            slope = slope[slope != 0]
            turns, last = 0, None
            ext = [sm.iloc[0]]
            for i in range(1, len(slope)):
                if slope[i] != slope[i - 1]:
                    v = sm.iloc[i]
                    if abs(v - ext[-1]) >= 0.05:
                        turns += 1
                        ext.append(v)
            slot_idx = np.round((w["eps_u"] - w["slot_err_deg"]) / S)
            a0, a1 = w["slot_err_deg"].abs().iloc[:7].median(), w["slot_err_deg"].abs().iloc[-7:].median()
            if slot_idx.nunique() > 1:
                c = "migrating"
            elif turns >= 1:
                c = "reversing"
            elif a0 - a1 >= 0.3 and a1 <= 1:
                c = "approaching"
            elif sm.max() - sm.min() < 0.1:
                c = "stationary"
            else:
                c = "drifting"
            seg_rows.append(dict(sid=sid, age_start=float(g.set_index("day")["age_d"].reindex([s0]).fillna(np.nan).iloc[0]) if s0 in g["day"].values else np.nan,
                                 cls=c, on_grid=bool(w["slot_err_deg"].abs().median() <= 1)))
    SG = pd.DataFrame(seg_rows)
    SG["age_bin"] = pd.cut(SG["age_start"], [-1, 30, 90, 10000], labels=["0-30 d", "30-90 d", ">= 90 d"])
    tab = pd.crosstab(SG["age_bin"], SG["cls"], normalize="index").round(3)
    tab["segments"] = SG.groupby("age_bin", observed=False).size()
    add("- 30-day trajectory segments by class (share of segments, by time since arrival at segment start):")
    add("")
    add(md(tab.reset_index()))
    add("")
    add(f"- All segments: {SG['cls'].value_counts(normalize=True).round(3).to_dict()} (n = {len(SG)})")
    has = RV[RV["reversals"] >= 2]
    add(f"- Satellites with >= 60 d of record >= 90 d after arrival: {len(RV)}; with >= 2 turning points (7-day smoothed eps, "
        f">= 0.05 deg between turns): {len(has)} ({len(has) / len(RV):.0%})")
    add(f"- Among them: median interval between turning points {has['gap_med'].median():.0f} d (satellite medians, P10-P90 "
        f"{has['gap_med'].quantile(.1):.0f}-{has['gap_med'].quantile(.9):.0f}); median change of eps between turning points "
        f"{has['amp_med'].median():.2f} deg (P10-P90 {has['amp_med'].quantile(.1):.2f}-{has['amp_med'].quantile(.9):.2f})")
    add(f"- Range of the unwrapped eps over the steady-state record: median {MG['range_deg'].median():.2f} deg, P90 {MG['range_deg'].quantile(.9):.2f}; "
        f"satellites whose assigned slot changes after 90 d: {(MG['migrations'] > 0).sum()} of {len(MG)}")
    Pg = P[P["good"] & P["std_angle"]]
    add(f"- Relative pair drift (primary pair-windows): median {Pg['drift'].median():.1f} km/day, P90 {Pg['drift'].quantile(.9):.1f}")
    add("")

    # figure: representative satellites (most turning points; a late arrival approaching its slot; widest range)
    pick = []
    if len(has):
        pick.append(has.sort_values("reversals", ascending=False)["sid"].iloc[0])
        pick.append(has.sort_values("amp_med")["sid"].iloc[len(has) // 2])
    late = E[E["arrival"] >= E["arrival"].max() - pd.Timedelta(days=90)].groupby("NORAD_CAT_ID")["abs"].first().sort_values()
    if len(late):
        pick.append(late.index[-1])
    pick.append(MG.sort_values("range_deg")["sid"].iloc[-1])
    pick = list(dict.fromkeys(pick))
    # Figure 8 is drawn by 27_fig8_trajectories.py; the four satellites are chosen by the rules above and recorded here.
    reason = {pick[0]: "most turning points"} if len(has) else {}
    if len(has):
        reason.setdefault(has.sort_values("amp_med")["sid"].iloc[len(has) // 2], "median turning-point amplitude")
    if len(late):
        reason.setdefault(late.index[-1], "arrived in the last 90 days of the record, largest first-day |eps|")
    reason.setdefault(MG.sort_values("range_deg")["sid"].iloc[-1], "widest range of unwrapped eps")
    pd.DataFrame([dict(NORAD_CAT_ID=k, rule=v) for k, v in reason.items()]).to_csv(OUT / "fig8_picks.csv", index=False)
    add(f"- Figure 8 satellites (drawn by 27_fig8_trajectories.py; selection rules in out/fig8_picks.csv): "
        f"{', '.join(E[E['NORAD_CAT_ID'].isin(pick)].drop_duplicates('NORAD_CAT_ID')['OBJECT_ID'])}")
    add("")

    # ---------------- E. bridge to exposure
    add("## 6.4 Pair phase deviation and exposure (primary pair-windows)")
    Em = E.assign(t0=E["t"] - pd.Timedelta(hours=12)).set_index(["t0", "OBJECT_ID"])["eps_um"]
    Pg = Pg.assign(slot_rel=Em.reindex(pd.MultiIndex.from_arrays([Pg["t0"], Pg["a"]])).to_numpy()
                   - Em.reindex(pd.MultiIndex.from_arrays([Pg["t0"], Pg["b"]])).to_numpy()).dropna(subset=["slot_rel"])
    Pg = Pg.assign(shift_km=np.radians(Pg["slot_rel"].abs()) * R_KM * np.cos(np.radians(Pg["gamma"]) / 2),
                   dd=Pg["d_win"] - Pg["nominal"], closer=Pg["d_win"] < Pg["nominal"])
    qd = Pg["slot_rel"].abs().quantile([.5, .9, .99])
    add(f"- Pair-level phase deviation eps_pair = eps_a - eps_b (= Delta phi_observed - Delta phi_nominal): |eps_pair| median {qd[.5]:.2f} deg, "
        f"P90 {qd[.9]:.2f}, P99 {qd[.99]:.2f}; robust sigma of signed eps_pair {1.4826 * (Pg['slot_rel'] - Pg['slot_rel'].median()).abs().median():.2f} deg")
    q = Pg["shift_km"].quantile([.5, .9, .99])
    add(f"- Pair phase deviation as along-track distance at the crossing, |eps_a - eps_b| r cos(gamma/2): median {q[.5]:.1f} km, "
        f"P90 {q[.9]:.1f}, P99 {q[.99]:.1f} (all {len(Pg):,} primary pair-windows)")
    add("")
    nb = [0, 10, 20, 40, 60, 100, 200, np.inf]
    Pg = Pg.assign(nbin=pd.cut(Pg["nominal"], nb, right=False))
    rows = []
    for b, g in Pg.groupby("nbin", observed=True):
        ev, ne = g[g["obs10"]], g[~g["obs10"]]
        rows.append(dict(nominal_bin=str(b), pair_windows=len(g), events=len(ev), rate_per_1000=round(1000 * len(ev) / len(g), 2),
                         ev_shift_med_km=round(ev["shift_km"].median(), 1) if len(ev) else np.nan,
                         nonev_shift_med_km=round(ne["shift_km"].median(), 1),
                         ev_closer_than_nominal=round(ev["closer"].mean(), 3) if len(ev) else np.nan,
                         nonev_closer_than_nominal=round(ne["closer"].mean(), 3),
                         ev_dd_med_km=round(ev["dd"].median(), 1) if len(ev) else np.nan,
                         nonev_dd_med_km=round(ne["dd"].median(), 1)))
    add("")
    add(md(pd.DataFrame(rows)))
    add("")
    add("dd = window-model minimum with observed phases minus nominal margin; closer = share with dd < 0 (phase deviations reduce "
        "the separation below the nominal margin).")
    ev = Pg[Pg["obs10"]]
    add(f"- All events: {ev['closer'].mean():.1%} have window minimum below their nominal margin; median dd {ev['dd'].median():.1f} km "
        f"(n = {len(ev)}); events with nominal margin < 10 km: {ev[ev['nominal'] < 10]['closer'].mean():.1%} closer than nominal")

    (ROOT / "PHASE_CLAIMS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
