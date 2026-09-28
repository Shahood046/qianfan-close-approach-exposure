"""(A) Where the analytical crossing model departs from SGP4, term by term.
(B) Do phase drifts predict the observed V-shaped episodes?

(A) For the inter-plane pairs with SGP4 d_min <= 50 km on the reference date, add physics in steps:
    V1 circular orbits, common inclination and radius            (pure phase term)
    V2 + actual inclination of each satellite (osculating)
    V3 + actual semi-major axes
    V4 + eccentricity vectors (mean e, omega)                    (07_crossing_model.py)
    V5 + osculating SGP4 radii at the crossing instead of mean-element radii
    and report median/90th-percentile |model - SGP4|.

(B) For the operational pairs that came within 5 km on >= 5 days of the 30-day daily screen
    (encounter_pairs_daily.csv): compute the model d(t) each day from that day's element sets,
    compare with the screened daily d_min, then fit x(t) = r*dphi*cos(gamma/2) linearly on the first
    --fit-days days and predict the epoch of zero phase and the minimum distance for the rest.
    Independent check of the drift: dx/dt against r*cos(gamma/2)*(n1 - n2) from mean motions.

Usage:  py 09_model_checks.py [--fit-days 8]
"""
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import pandas as pd

from common import DATA, MU, OUT, load_gp, load_states, satrec, to_jd

_spec = importlib.util.spec_from_file_location("rb", OUT.parent / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
cm, pl = rb.cm, rb.pl


def crossings(a, b):
    """Both plane crossings for satellite arrays a, b: yield (gamma, u*_a, u*_b) per sign."""
    h1, h2 = cm.unit_h(a["raan"].to_numpy(), a["inc"].to_numpy()), cm.unit_h(b["raan"].to_numpy(), b["inc"].to_numpy())
    line = np.cross(h1, h2)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    gamma = np.degrees(np.arccos(np.clip((h1 * h2).sum(1), -1, 1)))
    for sgn in (1, -1):
        p = sgn * line
        yield gamma, cm.u_of_direction(p, a["raan"].to_numpy(), a["inc"].to_numpy()), cm.u_of_direction(p, b["raan"].to_numpy(), b["inc"].to_numpy())


def variant_d(el, iu, ju, radii=None, mean_phase=False):
    """Model distance; radii(sgn_index, us1, us2, iu, ju) -> (r1, r2) overrides the mean-element radii."""
    a, b = el.iloc[iu].reset_index(drop=True), el.iloc[ju].reset_index(drop=True)
    best = np.full(len(a), np.inf)
    for k, (gamma, us1, us2) in enumerate(crossings(a, b)):
        if mean_phase:
            def bar(u, e, w):
                return u - np.degrees(2 * e * np.sin(np.radians(u - w)))
            ea, wa, eb, wb = a["ECCENTRICITY"].to_numpy(), a["ARG_OF_PERICENTER"].to_numpy(), b["ECCENTRICITY"].to_numpy(), b["ARG_OF_PERICENTER"].to_numpy()
            dphi = np.radians(cm.wrap((bar(a["u"].to_numpy(), ea, wa) - bar(us1, ea, wa)) - (bar(b["u"].to_numpy(), eb, wb) - bar(us2, eb, wb))))
        else:
            dphi = np.radians(cm.wrap((a["u"].to_numpy() - us1) - (b["u"].to_numpy() - us2)))
        if radii is None:
            r1 = a["A_KM"].to_numpy() * (1 - a["ECCENTRICITY"].to_numpy() * np.cos(np.radians(us1 - a["ARG_OF_PERICENTER"].to_numpy())))
            r2 = b["A_KM"].to_numpy() * (1 - b["ECCENTRICITY"].to_numpy() * np.cos(np.radians(us2 - b["ARG_OF_PERICENTER"].to_numpy())))
        else:
            r1, r2 = radii(k, us1, us2, iu, ju)
        d = np.sqrt((0.5 * (r1 + r2) * dphi * np.cos(np.radians(gamma) / 2)) ** 2 + (r1 - r2) ** 2)
        best = np.minimum(best, d)
    return best


def sgp4_radius(sat, jd, fr, u_now, u_star, a_km):
    n = np.sqrt(MU / a_km**3)
    dt = (((u_star - u_now) % 360) * np.pi / 180) / n
    _, r, _ = sat.sgp4(jd, fr + dt / 86400)
    return np.linalg.norm(r)


def part_a(df, states, t0):
    el = rb.get_el(df, states, t0)
    near = df.assign(_dt=(df["EPOCH"] - t0).abs())
    near = near[near["_dt"] <= pd.Timedelta(hours=48)]
    near = near.loc[near.groupby("NORAD_CAT_ID")["_dt"].idxmin()].merge(
        states[["NORAD_CAT_ID", "EPOCH", "STATE"]], on=["NORAD_CAT_ID", "EPOCH"])
    op = near[near["STATE"] == "operational"].reset_index(drop=True)
    sats = [satrec(r) for _, r in op.iterrows()]
    jd, fr = to_jd(t0)

    obs = pd.read_csv(OUT / "ideal_vs_observed_pairs.csv")
    both = pd.concat([obs, obs.rename(columns={"id_1": "id_2", "id_2": "id_1"})])
    idx = {o: i for i, o in enumerate(el["OBJECT_ID"])}
    sel = obs[obs["observed_dmin_km"] <= 50]
    iu = np.array([idx[a] for a in sel["id_1"]]); ju = np.array([idx[b] for b in sel["id_2"]])
    truth = sel["observed_dmin_km"].to_numpy()

    e1 = el.assign(inc=el["inc"].mean(), ECCENTRICITY=0.0, A_KM=el["A_KM"].mean())
    e2 = el.assign(ECCENTRICITY=0.0, A_KM=el["A_KM"].mean())
    e3 = el.assign(ECCENTRICITY=0.0)

    def osc(k, us1, us2, iu_, ju_):
        r1 = np.array([sgp4_radius(sats[i], jd[0], fr[0], el["u"].iloc[i], u, el["A_KM"].iloc[i]) for i, u in zip(iu_, us1)])
        r2 = np.array([sgp4_radius(sats[j], jd[0], fr[0], el["u"].iloc[j], u, el["A_KM"].iloc[j]) for j, u in zip(ju_, us2)])
        return r1, r2

    versions = {"V1 circular, common i and a": variant_d(e1, iu, ju),
                "V2 + actual inclinations": variant_d(e2, iu, ju),
                "V3 + actual semi-major axes": variant_d(e3, iu, ju),
                "V4 + eccentricity vectors (mean e, omega)": variant_d(el, iu, ju),
                "V5 + osculating SGP4 radii at crossing": variant_d(el, iu, ju, radii=osc),
                "V6 = V4 with phase in mean argument of latitude": variant_d(el, iu, ju, mean_phase=True)}
    save = pd.DataFrame({"id_1": sel["id_1"].to_numpy(), "id_2": sel["id_2"].to_numpy(), "sgp4_dmin_km": truth})
    for name, d in versions.items():
        save[name.split()[0]] = d
    save["gamma_deg"] = [next(iter(crossings(el.iloc[[i]], el.iloc[[j]])))[0][0] for i, j in zip(iu, ju)]
    order = el.groupby("plane")["raan"].apply(lambda x: pl.circmean(x.to_numpy(), 360)).sort_values().index    # P1.. by RAAN (as 17_geometry.py)
    lab = {p_: f"P{i + 1}" for i, p_ in enumerate(order)}
    save["plane_1"] = el["plane"].map(lab).to_numpy()[iu]
    save["plane_2"] = el["plane"].map(lab).to_numpy()[ju]
    save.to_csv(OUT / "model_vs_sgp4_pairs.csv", index=False)
    print(f"A. model vs SGP4, {len(sel)} inter-plane pairs with d_min <= 50 km, {t0:%Y-%m-%d %H:%M}")
    rows = []
    for name, d in versions.items():
        err = np.abs(d - truth)
        rows.append((name, np.corrcoef(np.log(d + 0.1), np.log(truth + 0.1))[0, 1], np.median(err), np.percentile(err, 90),
                     np.median(np.abs(d - truth) / (truth + 1))))
    print(pd.DataFrame(rows, columns=["version", "r(log d)", "median |err| km", "p90 |err| km", "median rel err"]).round(3).to_string(index=False))
    worst = pd.DataFrame({"id_1": sel["id_1"].to_numpy(), "id_2": sel["id_2"].to_numpy(), "sgp4": truth,
                          "V4": versions["V4 + eccentricity vectors (mean e, omega)"],
                          "gamma": [next(iter(crossings(el.iloc[[i]], el.iloc[[j]])))[0][0] for i, j in zip(iu, ju)]})
    worst["err"] = (worst["V4"] - worst["sgp4"]).abs()
    print("\n   largest V4 errors:\n" + worst.nlargest(6, "err").round(2).to_string(index=False) + "\n")


def part_b(df, states, fit_days):
    pr = pd.read_csv(OUT / "encounter_pairs_daily.csv", parse_dates=["t0"])
    pr = pr[(pr["pair_type"] == "QQ_diff_launch") & (pr["STATE_1"] == "operational") & (pr["STATE_2"] == "operational")]
    pr = pr.assign(day=pr["t0"].dt.tz_convert(None).dt.floor("D"))
    days = sorted(pr["day"].unique())
    close = pr[pr["d_min_km"] <= 5].groupby(["OBJECT_ID_1", "OBJECT_ID_2"])["day"].nunique()
    tracked = close[close >= 5].index.tolist()
    print(f"B. {len(tracked)} operational pairs within 5 km on >= 5 of {len(days)} days")

    series = {p: [] for p in tracked}
    for d in days:
        t = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=12)
        el = rb.get_el(df, states, t)
        idx = {o: i for i, o in enumerate(el["OBJECT_ID"])}
        for (a, b) in tracked:
            if a not in idx or b not in idx:
                series[(a, b)].append((d, np.nan, np.nan, np.nan, np.nan)); continue
            i, j = np.array([idx[a]]), np.array([idx[b]])
            A, B = el.iloc[i].reset_index(drop=True), el.iloc[j].reset_index(drop=True)
            best = None
            for k, (gamma, us1, us2) in enumerate(crossings(A, B)):
                ea, wa = A["ECCENTRICITY"].to_numpy(), A["ARG_OF_PERICENTER"].to_numpy()
                eb, wb = B["ECCENTRICITY"].to_numpy(), B["ARG_OF_PERICENTER"].to_numpy()
                dphi = np.radians(cm.wrap((cm.mean_phase(A["u"].to_numpy(), ea, wa) - cm.mean_phase(us1, ea, wa))
                                          - (cm.mean_phase(B["u"].to_numpy(), eb, wb) - cm.mean_phase(us2, eb, wb))))[0]
                r1 = A["A_KM"].iloc[0] * (1 - A["ECCENTRICITY"].iloc[0] * np.cos(np.radians(us1[0] - A["ARG_OF_PERICENTER"].iloc[0])))
                r2 = B["A_KM"].iloc[0] * (1 - B["ECCENTRICITY"].iloc[0] * np.cos(np.radians(us2[0] - B["ARG_OF_PERICENTER"].iloc[0])))
                x = 0.5 * (r1 + r2) * dphi * np.cos(np.radians(gamma[0]) / 2)
                if best is None or abs(x) < abs(best[0]):
                    best = (x, r1 - r2, k, (A["A_KM"].iloc[0], B["A_KM"].iloc[0], gamma[0]))
            series[(a, b)].append((d, best[0], best[1], best[2], best[3]))

    rows = []
    for (a, b), s in series.items():
        S = pd.DataFrame(s, columns=["day", "x", "dr", "crossing", "info"]).dropna(subset=["x"])
        obs = pr[(pr["OBJECT_ID_1"] == a) & (pr["OBJECT_ID_2"] == b)].groupby("day")["d_min_km"].min().reindex(S["day"]).to_numpy()
        S["model"] = np.hypot(S["x"], S["dr"]); S["obs"] = obs
        S.assign(pair=f"{a}/{b}").drop(columns="info").to_csv(OUT / "episode_series.csv", mode="a", header=not (OUT / "episode_series.csv").exists(), index=False)
        ok = S.dropna(subset=["obs"])
        r = np.corrcoef(np.log(ok["model"] + 0.2), np.log(ok["obs"] + 0.2))[0, 1] if len(ok) > 4 else np.nan
        # linear drift fit on the first fit_days days
        fit = S.iloc[:fit_days]; t = np.arange(len(S), dtype=float)
        if len(fit) >= 5 and fit["crossing"].nunique() == 1:
            slope, icpt = np.polyfit(t[:len(fit)], fit["x"], 1)
        else:
            slope, icpt = np.nan, np.nan
        t_zero = -icpt / slope if slope and np.isfinite(slope) and slope != 0 else np.nan
        later = ok.iloc[fit_days:]
        t_obs = float(t[list(S.index).index(later["obs"].idxmin())]) if len(later) and later["obs"].notna().any() else np.nan
        A_, B_, gam = S["info"].iloc[0]
        n1, n2 = np.sqrt(MU / A_**3), np.sqrt(MU / B_**3)
        rc = 0.5 * (A_ + B_) * np.cos(np.radians(gam) / 2)
        slope_n = rc * (n1 - n2) * 86400            # km/day implied by mean-motion difference
        rows.append(dict(pair=f"{a}/{b}", days=len(S), r_log=r, obs_min=ok["obs"].min(), model_min=S["model"].min(),
                         slope_fit=slope, slope_from_dn=slope_n, t_zero_pred=t_zero, t_min_obs_after_fit=t_obs))
    R = pd.DataFrame(rows)
    print(R.round(2).to_string(index=False))
    v = R.dropna(subset=["slope_fit", "slope_from_dn"])
    if len(v):
        print(f"\n   model d(t) vs screened daily d_min: median r(log) = {R['r_log'].median():.2f} over {R['r_log'].notna().sum()} pairs")
        print(f"   drift slope: fitted vs from mean-motion difference (n={len(v)}): r = {np.corrcoef(v['slope_fit'], v['slope_from_dn'])[0, 1]:.2f}, "
              f"median ratio fitted/predicted = {(v['slope_fit'] / v['slope_from_dn']).median():.2f}")
    w = R.dropna(subset=["t_zero_pred", "t_min_obs_after_fit"])
    w = w[(w["t_zero_pred"] > 0) & (w["t_zero_pred"] < len(days) + 5)]
    if len(w):
        e = (w["t_zero_pred"] - w["t_min_obs_after_fit"])
        print(f"   predicted vs observed epoch of closest approach, pairs where it falls after the fit window (n={len(w)}): "
              f"median error {e.median():.1f} d, 90th pct |error| {e.abs().quantile(0.9):.1f} d")
    R.to_csv(OUT / "episode_prediction.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit-days", type=int, default=8)
    args = ap.parse_args()
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    part_a(df, states, df["EPOCH"].max().floor("h") - pd.Timedelta(days=1))
    part_b(df, states, args.fit_days)


if __name__ == "__main__":
    main()
