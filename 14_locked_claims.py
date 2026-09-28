"""Locked-claims table: every quantitative claim of paper 1, computed from the merged daily record.

Prerequisites
    py 04_screen.py --every-days 1 --start 2025-01-01 --end 2026-03-31 --kinds QQ --tag _dailyB
    (encounter_pairs_dailyA.csv already exists for 2026-04-01 .. 2026-09-22)
    merge:  concatenate encounter_pairs_dailyB.csv and encounter_pairs_dailyA.csv -> encounter_pairs_daily.csv
    py 11_per_angle.py --pairs encounter_pairs_daily.csv --tag _daily
Then:  py 14_locked_claims.py

Writes LOCKED_CLAIMS.md. Claim types: OBSERVED (reconstructed or measured from orbital data), MODEL (from the crossing
model or SGP4 screening), INTERPRETIVE (explanations; allowed wording listed separately).
Uncertainty: 7-day moving-block bootstrap over windows (adjacent daily windows are correlated) unless stated.
Subset unless stated: on-grid (both satellites within 1 deg of the slot grid), slowly drifting (< 50 km/day), standard angles.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from scipy.stats import chi2

from common import DATA, OUT, load_gp, load_states

_spec = importlib.util.spec_from_file_location("rb", OUT.parent / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
cm, pl = rb.cm, rb.pl

ROWS: list[dict] = []
BLOCK = 7
NBOOT = 300
EARLY_END = pd.Timestamp("2026-04-01", tz="UTC")


def add(claim, value, denom, period, method, unc, where, kind):
    ROWS.append(dict(Claim=claim, Value=value, Population=denom, Period=period, Method=method, Uncertainty=unc, Where=where, Type=kind))
    print(f"[{kind[:3]}] {claim}: {value}  | {denom} | {period} | {unc}", flush=True)


def poisson_ci(k, n, scale=1000):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0
    return scale * lo / n, scale * chi2.ppf(0.975, 2 * k + 2) / 2 / n


def share_at(nom, obs, fracs):
    o = np.argsort(nom, kind="stable")
    cum = np.cumsum(obs[o]) / max(obs.sum(), 1)
    n = len(nom)
    return np.array([cum[max(int(f * n) - 1, 0)] for f in fracs])


def block_boot(G, fracs, nb=NBOOT, block=BLOCK, seed=0):
    days = sorted(G["t0"].unique())
    nom = {d: g["nominal"].to_numpy() for d, g in G.groupby("t0")}
    obs = {d: g["obs10"].to_numpy() for d, g in G.groupby("t0")}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(len(days) / block))
    shares, rates = [], []
    for _ in range(nb):
        starts = rng.integers(0, max(len(days) - block, 1), nblk)
        sel = [days[s + j] for s in starts for j in range(block) if s + j < len(days)]
        nn = np.concatenate([nom[d] for d in sel])
        oo = np.concatenate([obs[d] for d in sel])
        shares.append(share_at(nn, oo, fracs))
        rates.append(1000 * oo.mean())
    return np.array(shares), np.array(rates)


def episodes(ev: pd.DataFrame, windows: list, gap: int = 1) -> pd.DataFrame:
    idx = {t: i for i, t in enumerate(windows)}
    ev = ev.assign(w=ev["t0"].map(idx)).sort_values("w")
    rows = []
    for (a, b), d in ev.groupby(["a", "b"]):
        run = []
        for r in d.itertuples():
            if run and r.w - run[-1].w > gap + 1:
                rows.append(run)
                run = []
            run.append(r)
        if run:
            rows.append(run)
    out = []
    for run in rows:
        j = int(np.argmin([r.d_obs for r in run]))
        out.append(dict(a=run[0].a, b=run[0].b, start=run[0].t0, end=run[-1].t0, days=(run[-1].w - run[0].w) + 1,
                        n_events=len(run), dmin=run[j].d_obs, nominal_at_min=run[j].nominal,
                        min_nominal=min(r.nominal for r in run), k=run[0].k, slot_max=run[j].slot_max))
    return pd.DataFrame(out)


def main():
    P = pd.read_pickle(DATA / "pair_windows_daily.pkl.gz")
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    P3 = pd.read_pickle(DATA / "pair_windows.pkl.gz")
    P3["t0"] = pd.to_datetime(P3["t0"], utc=True)
    states = load_states()
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    G = P[P["good"] & P["std_angle"]].copy()
    G["period"] = np.where(G["t0"] < EARLY_END, "early", "recent")
    windows = sorted(P["t0"].unique())
    t_first, t_last = pd.Timestamp(windows[0]), pd.Timestamp(windows[-1])
    span = f"{t_first:%Y-%m-%d} to {t_last:%Y-%m-%d} ({len(windows)} daily windows)"
    print(f"record: {span}; pair-windows {len(P):,}; subset {len(G):,}\n")

    # ---------------- A data
    last = states.sort_values("EPOCH").groupby("NORAD_CAT_ID").last()["STATE"].value_counts()
    n_launched = states["NORAD_CAT_ID"].nunique()
    n_op_last = int(P.loc[P["t0"] == P["t0"].max(), "n_op"].iloc[0])
    n_pl_last = int(P.loc[P["t0"] == P["t0"].max(), "n_planes"].iloc[0])
    add("Satellites with element-set history", n_launched, "Space-Track payloads named Qianfan (15 launches; launch 2026-211 not yet catalogued)", "2024-08 to 2026-09-24", "Space-Track gp_history", "count", "Data", "OBSERVED")
    add("Operational satellites at the end of the record", n_op_last, "retrospective state classifier (arrival within a +-15 km band for >= 30 d)", f"{t_last:%Y-%m-%d}", "state classification, 02_states.py", "X 10-30 km, K 14-30 d: arrival epochs move by a median 12 d (P90 24 d); arrived count 172-181 at K = 30 d (179-195 at K = 14 d)", "Data", "OBSERVED")
    add("Final states of all satellites", ", ".join(f"{k} {v}" for k, v in last.items()), f"{n_launched} satellites", "as of 2026-09-24", "02_states.py", "20 stranded vs 22 failed in an independent tracker", "Data", "OBSERVED")
    # ---------------- B/C geometry at final date
    t_end = t_last + pd.Timedelta(hours=12)
    el = rb.slots(rb.get_el(df, states, t_end), rb.ESTIMATORS["circular mean"])
    npl = el.groupby("plane").size()
    populated = npl[npl >= 5].index
    add("Plane positions at the end of the record", f"{n_pl_last}: {len(populated)} populated ({npl[populated].min()}-{npl[populated].max()} satellites) + {n_pl_last - len(populated)} sparse (3 surviving satellites of launch 2024-185)", "operational satellites", f"{t_last:%Y-%m-%d}", "RAAN clustering (2 deg gap); populated = >= 5 satellites", "any clustering gap between 0.86 and 18.5 deg gives the same planes (17_geometry.py)", "Geometry", "OBSERVED")
    pr = np.sort(el[el["plane"].isin(populated)].groupby("plane")["raan"].apply(lambda x: pl.circmean(x.to_numpy(), 360)).to_numpy())
    gaps = np.diff(np.r_[pr, pr[0] + 360])
    gaps = np.delete(gaps, gaps.argmax())                       # the gap outside the star
    steps = np.maximum(np.round(gaps / 20.5), 1)
    d = np.repeat(gaps / steps, steps.astype(int))
    add("RAAN spacing per lattice step between populated planes", f"median {np.median(d):.2f} deg (range {d.min():.2f}-{d.max():.2f}); planes span {360 - np.diff(np.r_[pr, pr[0] + 360]).max():.1f} deg", f"{len(populated)} populated planes (the sparse 2024-185 plane lies ~2 deg off the lattice)", f"{t_end:%Y-%m-%d}", "circular-mean RAAN per plane", "spread of individual satellites within a plane < 1 deg; 18-snapshot median 20.40-20.49 (17_geometry.py)", "Geometry", "OBSERVED")
    grid = np.arange(4, 25.01, 0.05)
    R = np.array([rb.coherence(el, s) for s in grid])
    strong = grid[(grid >= 8) & (R >= 0.9)]
    within1 = (el["slot_err_deg"].abs() <= 1).mean()
    add("In-plane slot spacing", f"10 deg: coherence R(10) = {R[np.argmin(abs(grid - 10))]:.3f}; R >= 0.9 only for {strong.min():.2f}-{strong.max():.2f} deg", f"{len(el)} operational satellites", f"{t_end:%Y-%m-%d}", "pooled circular coherence, free spacing scan 4-25 deg", "leave-one-out / estimator / bootstrap: nominal margins of closest pairs move by < 15 km (08_robustness.py)", "Geometry", "OBSERVED")
    add("Satellites within 1 deg of the empirical slot grid", f"{within1:.1%}", f"{len(el)} operational satellites", f"{t_end:%Y-%m-%d}", "mean argument of latitude", "n/a", "Geometry", "OBSERVED")
    # coherence over time
    rs = []
    for dt in pd.date_range(pd.Timestamp("2025-06-01", tz="UTC"), t_last, freq="90D"):
        try:
            e = rb.slots(rb.get_el(df, states, dt), rb.ESTIMATORS["circular mean"])
            rs.append((dt, len(e), rb.coherence(e, 10)))
        except ValueError:
            pass
    add("Grid coherence R(10) over time", "; ".join(f"{d:%Y-%m}: {r:.2f} (n={n})" for d, n, r in rs), "operational satellites", "2025-06 to 2026-09", "same estimator, every 90 d", "n/a", "Geometry", "OBSERVED")

    # ---------------- D satellite-level phase errors: daily table in SGP4 mean argument of latitude (19_phase_dynamics.py)
    Ed = pd.read_pickle(DATA / "slot_errors_daily.pkl.gz")
    Ed = Ed[Ed["LAUNCH"] != "2024-185"]
    a = Ed["eps_um"].abs()
    add("Absolute phase error, operational satellites of populated planes", f"median {a.median():.2f}, P90 {a.quantile(.9):.2f}, P95 {a.quantile(.95):.2f}, P99 {a.quantile(.99):.2f} deg; within 1 deg {(a <= 1).mean():.1%}", f"{len(Ed):,} satellite-days ({Ed['NORAD_CAT_ID'].nunique()} satellites)", span, "SGP4 mean argument of latitude (om + mm) vs each plane's circular-mean grid, daily", "satellite-days are not independent; the equation-of-centre phase gives nearly the same distribution (median 0.20) but adds a -0.114 deg cos u term that corrupts day-to-day changes (PHASE_CLAIMS.md)", "Phase", "OBSERVED")
    Ed = Ed.assign(t0=Ed["t"] - pd.Timedelta(hours=12)).set_index(["t0", "OBJECT_ID"])["eps_um"]
    rel = (Ed.reindex(pd.MultiIndex.from_arrays([G["t0"], G["a"]])).to_numpy() - Ed.reindex(pd.MultiIndex.from_arrays([G["t0"], G["b"]])).to_numpy())
    rel = rel[np.isfinite(rel)]
    sig = 1.4826 * np.median(np.abs(rel - np.median(rel)))
    add("Pair phase deviation eps_a - eps_b (= Delta phi observed - nominal)", f"robust sigma {sig:.2f} deg; |eps_pair| median {np.median(np.abs(rel)):.2f}, P90 {np.quantile(np.abs(rel), .9):.2f}, P99 {np.quantile(np.abs(rel), .99):.2f} deg", f"{len(rel):,} on-grid slow standard-angle pair-windows (sparse plane excluded)", span, "SGP4 mean phase, daily", "a single Gaussian does not fit (heavier centre)", "Phase", "OBSERVED")
    add("Persistence of a satellite's signed phase error (on grid at both dates, >= 90 d after arrival)", "r = 0.975 (1 d), 0.927 (7 d), 0.474 (30 d), -0.018 (60 d), -0.168 (90 d), 0.067 (180 d)", "same satellite, daily table", span, "Pearson r of signed eps_um at lag L (19_phase_dynamics.py)", "all satellites: 0.41 / 0.38 / 0.15 / -0.05 / -0.15 / 0.00; 30-d snapshot estimate with the equation-of-centre phase was 0.50 (30 d), 0.05 (90 d)", "Phase", "OBSERVED")

    # ---------------- E model accuracy
    cmp = pd.read_csv(OUT / "crossing_model_pairs.csv").dropna(subset=["observed_dmin_km", "d_model_km"])
    cmp = cmp[cmp["observed_dmin_km"] <= 50]
    err = (cmp["d_model_km"] - cmp["observed_dmin_km"]).abs()
    add("Crossing model vs full SGP4 propagation over one orbit", f"r = {np.corrcoef(np.log(cmp['d_model_km'] + .1), np.log(cmp['observed_dmin_km'] + .1))[0, 1]:.3f}; median |error| {err.median():.2f} km; P90 {err.quantile(.9):.2f} km", f"{len(cmp)} inter-plane pairs with SGP4 d_min <= 50 km", "2026-09-23 (single epoch)", "mean-phase model (V6) vs SGP4", "term-by-term decomposition V1-V6 in FINDINGS R5", "Model", "MODEL")
    g2 = P[P["good"] & P["std_angle"] & (P["mod10"] | P["obs10"])]
    tp, fp, fn = int((g2["mod10"] & g2["obs10"]).sum()), int((g2["mod10"] & ~g2["obs10"]).sum()), int((~g2["mod10"] & g2["obs10"]).sum())
    add("Model (24 h window minimum <= 10 km) vs SGP4 screen, pair-window level", f"recall {tp / (tp + fn):.1%}, precision {tp / (tp + fp):.1%}", f"{tp + fp + fn:,} pair-windows flagged by either, on-grid slow standard-angle", span, "24 h window with linear phase drift", "12 subset definitions: recall 94-95%, precision 93-94% (12_sensitivity.py, 3-day)", "Model", "MODEL")

    Gd = P[P["good"] & P["std_angle"] & P["d_obs"].notna()]
    H = Gd[Gd["d_obs"] <= 10]
    eh, eg = (H["d_win"] - H["d_obs"]).abs(), (Gd["d_win"] - Gd["d_obs"]).abs()
    Ou = P[P["std_angle"] & P["d_obs"].notna() & ~P["good"]]
    eo = (Ou["d_win"] - Ou["d_obs"]).abs()
    add("24 h window model vs SGP4 screen minimum, pair-window level", f"events (SGP4 <= 10 km): median |error| {eh.median():.2f} km, P90 {eh.quantile(.9):.2f} km, median bias {(H['d_win'] - H['d_obs']).median():+.2f} km; all screened <= 20 km: median {eg.median():.2f}, P90 {eg.quantile(.9):.2f} km", f"{len(H):,} / {len(Gd):,} on-grid slow standard-angle pair-windows", span, "window minimum with linear phase drift vs SGP4 screen (10 s grid, refined)", f"outside the primary population (off-grid or fast drift): median {eo.median():.1f} km, P90 {eo.quantile(.9):.0f} km (n = {len(Ou):,}) -> validation domain only", "Model", "MODEL")
    pr = pd.read_csv(OUT / "encounter_pairs_daily.csv")
    pr["t0"] = pd.to_datetime(pr["t0"], utc=True)
    pr = pr.assign(a=np.minimum(pr["OBJECT_ID_1"], pr["OBJECT_ID_2"]), b=np.maximum(pr["OBJECT_ID_1"], pr["OBJECT_ID_2"]))
    mv = Gd.merge(pr[["t0", "a", "b", "v_rel_kms", "abs_lat_deg"]], on=["t0", "a", "b"])
    vm = 2 * np.sqrt(cm.MU / (6378.135 + 1068.5)) * np.sin(np.radians(mv["gamma"]) / 2)
    dv = (vm - mv["v_rel_kms"]).abs()
    add("Relative speed at closest approach: 2 v sin(gamma/2) vs SGP4", f"median |difference| {dv.median():.3f} km/s, P90 {dv.quantile(.9):.3f}, max {dv.max():.3f} (observed range {mv['v_rel_kms'].min():.1f}-{mv['v_rel_kms'].max():.1f} km/s); |latitude| at TCA median {mv['abs_lat_deg'].median():.1f} deg", f"{len(mv):,} primary pair-windows with SGP4 d_min <= 20 km", span, "median over passes in the window", "n/a", "Model", "MODEL")

    # ---------------- F rates
    for name, X in (("pooled", G), ("early (before 2026-04)", G[G["period"] == "early"]), ("recent (2026-04 onward)", G[G["period"] == "recent"])):
        k, n = int(X["obs10"].sum()), len(X)
        lo, hi = poisson_ci(k, n)
        add(f"Pair-window event rate, {name}", f"{1000 * k / n:.3f} per 1000 (Poisson CI {lo:.3f}-{hi:.3f})", f"{k} events / {n:,} on-grid slow standard-angle pair-windows", span if name == "pooled" else name, "daily cadence, SGP4 screen d_min <= 10 km in a 24 h window", "Poisson only (ignores dependence between windows, so too narrow; block bootstrap is reported for the concentration)", "Results", "MODEL")
    Pu = P[P["std_angle"]]
    add("Pair-window event rate, no on-grid / drift restriction", f"{1000 * Pu['obs10'].mean():.3f} per 1000", f"{int(Pu['obs10'].sum())} events / {len(Pu):,} pair-windows", span, "daily", "n/a", "Results", "MODEL")

    # ---------------- G concentration
    fracs = np.array([0.005, 0.01, 0.015, 0.02])
    point = {"pooled": share_at(G["nominal"].to_numpy(), G["obs10"].to_numpy(), fracs)}
    boots = {"pooled": block_boot(G, fracs)}
    for per in ("early", "recent"):
        X = G[G["period"] == per]
        point[per] = share_at(X["nominal"].to_numpy(), X["obs10"].to_numpy(), fracs)
        boots[per] = block_boot(X, fracs, seed=1)
    for per in ("pooled", "early", "recent"):
        sh, _ = boots[per]
        lo, hi = np.percentile(sh[:, 1], [2.5, 97.5])
        X = G if per == "pooled" else G[G["period"] == per]
        add(f"Events captured by the 1% of pair-windows with the smallest nominal margin, {per}", f"{point[per][1]:.1%} (0.5%: {point[per][0]:.1%}; 1.5%: {point[per][2]:.1%})", f"{int(X['obs10'].sum())} events / {len(X):,} pair-windows", span if per == "pooled" else per, "rank-based, daily cadence, on-grid slow standard-angle", f"7-day moving-block bootstrap, 95% interval {lo:.1%}-{hi:.1%} at 1%", "MAIN RESULT", "MODEL")
    # cadence comparison for the overlap period
    ov = G[G["t0"] >= EARLY_END]
    G3 = P3[P3["good"] & P3["std_angle"] & (P3["t0"] >= EARLY_END)]
    c_d = share_at(ov["nominal"].to_numpy(), ov["obs10"].to_numpy(), [0.01])[0]
    c_3 = share_at(G3["nominal"].to_numpy(), G3["obs10"].to_numpy(), [0.01])[0]
    add("Cadence check of the 1% concentration (2026-04 onward)", f"daily {c_d:.1%} vs 3-day {c_3:.1%}", f"{int(ov['obs10'].sum())} vs {int(G3['obs10'].sum())} events", "2026-04-01 to 2026-09-22", "same definition, different sampling", "n/a", "Cadence", "MODEL")
    # quarterly
    qrows, qn, qv = [], [], []
    G["q"] = G["t0"].dt.tz_convert(None).dt.to_period("Q").astype(str)
    for qq, X in G.groupby("q"):
        if X["obs10"].sum() >= 5:
            v = share_at(X['nominal'].to_numpy(), X['obs10'].to_numpy(), [0.01])[0]
            qrows.append(f"{qq}: {v:.0%} (n={int(X['obs10'].sum())})")
            qn.append(int(X['obs10'].sum())); qv.append(v)
            qv15 = share_at(X['nominal'].to_numpy(), X['obs10'].to_numpy(), [0.015])[0]
            qrows[-1] += f" [1.5%: {qv15:.0%}]"
    add("1% concentration by quarter (each quarter ranked separately)", "; ".join(qrows), "events per quarter as shown", span, "rank-based, daily", f"1% range {min(qv):.0%}-{max(qv):.0%} (1.5% column in brackets: every quarter 96-100%); {min(qn)}-{max(qn)} events per quarter; the lowest values (2025Q3, 2026Q3) occur where the quarter's own 1% cut slices through a plane-pair class (2026Q3: 92.7% at 1%, 98.3% at 1.25%)", "Results", "MODEL")
    # 60 km point
    near = G["nominal"] < 60
    def sh(X, th):
        return (X["obs10"] & (X["nominal"] < th)).sum() / X["obs10"].sum()
    Ge, Gr = G[G["period"] == "early"], G[G["period"] == "recent"]
    add("60 km point on the curve (secondary)", f"{(G['obs10'] & near).sum() / G['obs10'].sum():.1%} of events in {near.mean():.2%} of pair-windows", "on-grid slow standard-angle", span, "threshold 60 km", f"threshold-dependent: early {sh(Ge, 60):.0%} at 60 km / {sh(Ge, 80):.0%} at 80 km / {sh(Ge, 100):.0%} at 100 km; recent {sh(Gr, 60):.0%} / {sh(Gr, 80):.0%} / {sh(Gr, 100):.0%}. Do not headline", "Results", "MODEL")

    # unrestricted, distinct-pair and threshold views of the same concentration (Section 7.4)
    U = P[P["std_angle"]]
    su = share_at(U["nominal"].to_numpy(), U["obs10"].to_numpy(), [0.01, 0.05])
    Ou = U[~U["good"] & U["obs10"]]
    add("Concentration without the on-grid / slow-drift restriction", f"smallest 1% of pair-windows: {su[0]:.1%} of events; smallest 5%: {su[1]:.1%}", f"{int(U['obs10'].sum())} events / {len(U):,} standard-angle pair-windows", span, "rank-based, daily", f"{len(Ou)} events outside the primary population: {(Ou['slot_max'] > 1).mean():.0%} involve a satellite > 1 deg off grid, {(Ou['drift'] >= 50).mean():.0%} a pair drifting >= 50 km/day", "Results", "MODEL")
    cut = np.quantile(G["nominal"], 0.01)
    pm = G.groupby(["a", "b"]).agg(minnom=("nominal", "min"), ev=("obs10", "any"))
    add("Concentration in distinct pairs", f"{int((pm['minnom'] < 100).sum())} of {len(pm):,} distinct pairs ({(pm['minnom'] < 100).mean():.1%}) ever have nominal margin < 100 km; {int((pm['ev'] & (pm['minnom'] < 100)).sum())} of the {int(pm['ev'].sum())} pairs with an event are among them; the 1% pair-window cut is at a nominal margin of {cut:.0f} km", "primary population", span, "daily", "n/a", "Results", "MODEL")

    # ---------------- H episodes
    ev = G[G["obs10"]]
    ep = episodes(ev, windows)
    ep_all = episodes(P[P["obs10"] & P["std_angle"]], windows)
    one = (ep["days"] == 1).mean()
    add("Daily episodes (primary subset)", f"{len(ep)} episodes among {ep.groupby(['a', 'b']).ngroups} distinct pairs", "on-grid slow standard-angle event pair-windows, gap <= 1 day", span, "daily", "arrival band X = 10-30 km: 248-289 episodes (BANDWIDTH_CHECK.md); pair-window results change little", "Episodes", "MODEL")
    add("Episode duration (days, daily resolution)", f"median {ep['days'].median():.0f}, P90 {ep['days'].quantile(.9):.0f}, max {ep['days'].max()}; one-day episodes {one:.0%}", f"{len(ep)} episodes", span, "consecutive days with an event, one missed day allowed", "durations are lower bounds at window edges", "Episodes", "MODEL")
    rec = ep.groupby(["a", "b"]).size()
    multi = rec[rec > 1]
    gaps = []
    for (a, b), _ in multi.items():
        st = ep[(ep["a"] == a) & (ep["b"] == b)]["start"].sort_values().to_numpy()
        gaps += list(np.diff(st) / np.timedelta64(1, "D"))
    add("Recurring pairs (more than one episode)", f"{len(multi)} of {len(rec)} pairs; max {rec.max()} episodes; median gap between episode starts {np.median(gaps):.0f} d" if gaps else "none", f"{len(rec)} distinct pairs with an event", span, "daily", "n/a", "Episodes", "MODEL")
    long_ep = ep[ep["days"] >= 3]
    add("Episodes with a pair at nominal margin < 100 km", f"{(ep['min_nominal'] < 100).mean():.1%} of all episodes; {(long_ep['min_nominal'] < 100).mean():.1%} of episodes lasting >= 3 days ({len(long_ep)})", f"{len(ep)} episodes", span, "minimum nominal margin over the episode", "episode unit weights one-day events equally with long ones, so it is lower than the pair-window concentration (98%); the two are different units", "Episodes", "MODEL")
    add("Episodes including fast-drifting or off-grid pairs (context)", f"{len(ep_all)} episodes (all operational inter-plane pairs at standard angles)", "all event pair-windows", span, "daily", "n/a", "Episodes", "MODEL")

    # ---------------- I counterfactual
    close = ep[ep["dmin"] < 3]
    rng = np.random.default_rng(5)
    pairs_close = close.groupby(["a", "b"])
    keys = list(pairs_close.groups.keys())
    boots_cf = []
    for _ in range(1000):
        pick = rng.integers(0, len(keys), len(keys))
        x = pd.concat([close.loc[pairs_close.groups[keys[i]]] for i in pick])
        boots_cf.append(((x["nominal_at_min"] > 10).mean(), (x["nominal_at_min"] > 20).mean(), x["nominal_at_min"].median()))
    bc = np.array(boots_cf)
    add("Counterfactual (slot errors removed): episodes with observed d_min < 3 km", f"nominal median {close['nominal_at_min'].median():.1f} km; {(close['nominal_at_min'] > 10).mean():.0%} > 10 km; {(close['nominal_at_min'] > 20).mean():.0%} > 20 km", f"{len(close)} episodes ({len(keys)} pairs)", span, "nominal margin at the episode's closest day, model with both slot errors removed", f"pair-cluster bootstrap 95%: > 10 km {np.percentile(bc[:, 0], 2.5):.0%}-{np.percentile(bc[:, 0], 97.5):.0%}; > 20 km {np.percentile(bc[:, 1], 2.5):.0%}-{np.percentile(bc[:, 1], 97.5):.0%}; 12 subset definitions (slot limit 0.5-2 deg x drift limit 20-100 km/day) give 49-62% (12_sensitivity.py, 3-day)", "Counterfactual", "MODEL")
    rem = (ep["nominal_at_min"] < 5).mean()
    rem_close = (close["nominal_at_min"] < 5).mean()
    add("Remaining < 5 km after removing slot errors", f"{rem:.0%} of all episodes; {rem_close:.0%} of episodes with observed d_min < 3 km", f"{len(ep)} / {len(close)} episodes", span, "nominal margin < 5 km", "n/a", "Counterfactual", "MODEL")
    add("Five closest pairs (2026-09-23): observed -> slot errors removed", "0.84-3.13 km -> 24-50 km (estimators 10-64 km: the circular median puts 2026-104M/2026-108P at 9.8 km; leave-one-out 23-56 km; bootstrap 5-95% about 8-72 km)", "5 pairs", "2026-09-23 (single epoch)", "08_robustness.py T2/T3/T5", "as stated", "Counterfactual", "MODEL")

    # ---------------- J angle classes
    for k in sorted(G["k"].unique()):
        X = G[G["k"] == k]
        e = int(X["obs10"].sum())
        lo, hi = poisson_ci(e, len(X))
        add(f"Angle-class rate k = {k} ({k * 20.5:.0f} deg)", f"{1000 * e / len(X):.3f} per 1000 (CI {lo:.3f}-{hi:.3f})", f"{e} events / {len(X):,} pair-windows", span, "daily, on-grid slow", "Poisson", "Results", "MODEL")
    ev_e = G[G["k"] % 2 == 0]["obs10"].mean() * 1000
    ev_o = G[G["k"] % 2 == 1]["obs10"].mean() * 1000
    add("Even vs odd plane steps", f"even {ev_e:.2f}, odd {ev_o:.3f} per 1000 (ratio {ev_e / ev_o:.0f})", "on-grid slow standard-angle pair-windows", span, "daily", "ratio 45-114 across 12 subset definitions (3-day)", "Results", "MODEL")

    # ---------------- write markdown
    T = pd.DataFrame(ROWS)
    lines = ["# Locked claims (auto-generated by 14_locked_claims.py)", "",
             f"Record: {span}. Subset unless stated: on-grid (both satellites within 1 deg of the empirical slot grid), slowly drifting (< 50 km/day), standard angles. "
             "Uncertainty: 7-day moving-block bootstrap unless stated.", ""]
    for kind, title in (("OBSERVED", "1. Observed (reconstructed or measured from orbital data)"), ("MODEL", "2. Model-derived (crossing model and SGP4 screening)")):
        lines += [f"## {title}", "", "| Claim | Value | Population / denominator | Period | Method / cadence | Uncertainty | Where used |", "|---|---|---|---|---|---|---|"]
        for r in T[T["Type"] == kind].itertuples():
            lines.append(f"| {r.Claim} | {r.Value} | {r.Population} | {r.Period} | {r.Method} | {r.Uncertainty} | {r.Where} |")
        lines.append("")
    lines += ["## 3. Interpretive (explanations; allowed wording)", "",
              "| Interpretation | Allowed wording | Evidence | Not allowed |", "|---|---|---|---|",
              "| Alternate planes are in phase at the poles | 'consistent with staggered inter-plane phasing as used in star constellations' | even/odd contrast of rates and nominal margins (observed/model); prior parity results (Liang 2021; Handley 2018) | any statement of Qianfan design intent |",
              "| Relative phase is actively managed | 'consistent with active phase keeping' | signed phase errors (SGP4 mean phase) of every steady-state satellite reverse direction: turning points a median 23 d apart (P10-P90 9-54), 0.14 deg between turns; persistence r = 0.93 (7 d), 0.47 (30 d), ~0 (60 d), -0.17 (90 d); pair phase deviations show no systematic offset toward alignment (component toward alignment over primary pair-windows with nominal margin < 100 km: mean -0.007 deg, 95% block bootstrap -0.019 to +0.006, positive in 49.2%; SIGNED_DEVIATION.md) | naming a control law, dead-band size or manoeuvre schedule; any claim of collision-avoidance intent |",
              "| Rise of the event rate in 2026 | 'associated with a change in exposure composition: even-step plane pairs with coinciding occupied slots grew from 11% to 35% of primary pair-windows (2025-03-13..2026-03-31 vs 2026-09), with a similar within-class rate (4.2 vs 3.9 per 1000); a transient within-class increase in April-May 2026 came from one plane pair (P1-P6) whose nominal margin decreased' | RISE_TEST.md (18_rise_test.py): Kitagawa decomposition P0->Sep 2026 composition +0.93 (block bootstrap +0.73 to +1.12), within -0.05 (-0.46 to +0.27); early->recent composition +0.63 (+0.35 to +0.81), within +0.39 (+0.10 to +0.71); Apr-May within +1.10 (+0.57 to +1.73), P1-P6 rate 4.2 -> 15.1 as its median daily minimum nominal margin fell 61 -> 6.6 km; second-launch satellites (2026-108, 2026-125) carry 16% of coinciding-class events in Jun-Aug and 42% in Sep 2026 | 'caused by' any single factor; attributing the rise to second launches alone; any statement of intent |",
              "| Design implication | 'the tolerance that preserves a nominal margin m is m / (r cos(gamma/2))' (re-derivation; cite Yun et al. 2023 [V] as precedent; Bi et al. 2025 does not support a tolerance rule) | model | claiming the first analytic tolerance rule |",
              "| Concentration | 'exposure is highly non-uniform across inter-plane pair classes' | rank-based curve | 'designed to' / 'intended' |"]
    (OUT.parent / "LOCKED_CLAIMS.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwritten LOCKED_CLAIMS.md")


if __name__ == "__main__":
    main()
