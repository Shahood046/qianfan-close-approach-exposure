"""Signed pair phase deviation relative to the direction of alignment (manuscript Section 6.4). Writes SIGNED_DEVIATION.md.

For a pair (a, b) at the crossing that sets its nominal margin, the model phase offset is
    Delta phi_obs = Delta phi_nom + (eps_a - eps_b)          (exact: u_b,slotted = u_b - eps, SGP4 mean-phase slot errors)
The sign of eps_pair = eps_a - eps_b depends only on which satellite is listed first, so it says nothing about alignment.
The quantity that does is the component toward alignment
    toward = -sign(Delta phi_nom) * (eps_a - eps_b)          (deg; > 0: the deviation reduces the along-track offset)
    toward_km = toward * r cos(gamma/2)
Computed for every primary pair-window with nominal margin < 200 km and a 3% random sample of the rest (weighted back).
Uncertainty: 7-day moving-block bootstrap over windows.

Usage:  py 21_signed_deviation.py
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from common import DATA, RE, ROOT, load_gp, load_states

_spec = importlib.util.spec_from_file_location("rb", ROOT / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
cm = rb.cm
R_KM = RE + 1068.5
CACHE = DATA / "signed_deviation.pkl.gz"
SAMPLE = 0.03


def signed_at_nominal_crossing(el: pd.DataFrame, iu: np.ndarray, ju: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sign(Delta phi_nom) and eps_iu - eps_ju at the crossing where the slotted (nominal) distance is smallest."""
    a, b = el.iloc[iu].reset_index(drop=True), el.iloc[ju].reset_index(drop=True)
    h1, h2 = cm.unit_h(a["raan"].to_numpy(), a["inc"].to_numpy()), cm.unit_h(b["raan"].to_numpy(), b["inc"].to_numpy())
    line = np.cross(h1, h2)
    line /= np.linalg.norm(line, axis=1, keepdims=True)
    gamma = np.degrees(np.arccos(np.clip((h1 * h2).sum(1), -1, 1)))
    ea, wa, eb, wb = (a["ECCENTRICITY"].to_numpy(), a["ARG_OF_PERICENTER"].to_numpy(),
                      b["ECCENTRICITY"].to_numpy(), b["ARG_OF_PERICENTER"].to_numpy())
    best, sgn_nom = np.full(len(a), np.inf), np.zeros(len(a))
    for sgn in (1, -1):
        p = sgn * line
        us1 = cm.u_of_direction(p, a["raan"].to_numpy(), a["inc"].to_numpy())
        us2 = cm.u_of_direction(p, b["raan"].to_numpy(), b["inc"].to_numpy())
        dnom = cm.wrap((a["ub_slotted"].to_numpy() - cm.mean_phase(us1, ea, wa)) - (b["ub_slotted"].to_numpy() - cm.mean_phase(us2, eb, wb)))
        r1 = a["A_KM"].to_numpy() * (1 - ea * np.cos(np.radians(us1 - wa)))
        r2 = b["A_KM"].to_numpy() * (1 - eb * np.cos(np.radians(us2 - wb)))
        d = np.hypot(0.5 * (r1 + r2) * np.radians(dnom) * np.cos(np.radians(gamma) / 2), r1 - r2)
        upd = d < best
        best = np.where(upd, d, best)
        sgn_nom = np.where(upd, np.sign(dnom), sgn_nom)
    return sgn_nom, (a["slot_err_deg"].to_numpy() - b["slot_err_deg"].to_numpy())


def build(P: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    G = P[P["good"] & P["std_angle"] & ~(P["a"].str.startswith("2024-185") | P["b"].str.startswith("2024-185"))]
    sel = G[(G["nominal"] < 200) | (rng.random(len(G)) < SAMPLE)].copy()
    sel["w"] = np.where(sel["nominal"] < 200, 1.0, 1 / SAMPLE)
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    out = []
    for n, (t0, g) in enumerate(sel.groupby("t0")):
        el = rb.slots(rb.get_el(df, states, pd.Timestamp(t0) + pd.Timedelta(hours=12)), rb.ESTIMATORS["circular mean"])
        idx = {o: i for i, o in enumerate(el["OBJECT_ID"])}
        iu, ju = g["a"].map(idx).to_numpy(), g["b"].map(idx).to_numpy()
        s, e = signed_at_nominal_crossing(el, iu.astype(int), ju.astype(int))
        out.append(g.assign(sign_nom=s, eps_pair=e))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1} windows", flush=True)
    X = pd.concat(out, ignore_index=True)
    X["toward"] = -X["sign_nom"] * X["eps_pair"]
    X["toward_km"] = np.radians(X["toward"]) * R_KM * np.cos(np.radians(X["gamma"]) / 2)
    X.to_pickle(CACHE)
    return X


def wstats(x: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(np.average(x, weights=w)), float(x[o][np.searchsorted(cw, 0.5)]), float(np.average(x > 0, weights=w))


def boot_mean(X: pd.DataFrame, col: str, nb: int = 400, block: int = 7, seed: int = 4) -> tuple[float, float]:
    days = sorted(X["t0"].unique())
    by = {d: (g[col].to_numpy(), g["w"].to_numpy()) for d, g in X.groupby("t0")}
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(len(days) / block))
    m = []
    for _ in range(nb):
        starts = rng.integers(0, max(len(days) - block, 1), nblk)
        sel = [days[s + j] for s in starts for j in range(block) if s + j < len(days)]
        x = np.concatenate([by[d][0] for d in sel])
        w = np.concatenate([by[d][1] for d in sel])
        m.append(np.average(x, weights=w))
    return tuple(np.percentile(m, [2.5, 97.5]))


def main():
    P = pd.read_pickle(DATA / "pair_windows_daily.pkl.gz")
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    X = pd.read_pickle(CACHE) if CACHE.exists() else build(P)
    lines = ["# Signed pair phase deviation toward alignment (auto-generated by 21_signed_deviation.py)", "",
             f"Primary pair-windows (sparse plane excluded): all {int((X['w'] == 1).sum()):,} with nominal < 200 km, plus "
             f"{int((X['w'] > 1).sum()):,} sampled at {SAMPLE:.0%} from the rest (weighted back). toward > 0: the pair phase deviation "
             "reduces the along-track offset at the nominal crossing.", ""]
    m, med, fp = wstats(X["eps_pair"].to_numpy(), X["w"].to_numpy())
    lines.append(f"- Raw signed eps_pair (a - b, arbitrary order): weighted mean {m:+.3f} deg, median {med:+.3f}, share > 0 {fp:.1%}")
    rows = []
    nb = [0, 10, 20, 40, 60, 100, 200, np.inf]
    X["nbin"] = pd.cut(X["nominal"], nb, right=False)
    # Unconditional rows are the clean test (the nominal margin excludes the deviation); rows split by event outcome are
    # conditioned on the deviation itself and are shown only for reference.
    groups = [("all primary", X), ("nominal < 100 km, all outcomes", X[X["nominal"] < 100])]
    for (b, g) in X.groupby("nbin", observed=True):
        groups.append((f"all outcomes, nominal {b}", g))
    groups += [("[conditioned] events", X[X["obs10"]]), ("[conditioned] non-events", X[~X["obs10"]])]
    for (b, g) in X.groupby("nbin", observed=True):
        groups.append((f"[conditioned] non-events, nominal {b}", g[~g["obs10"]]))
    for name, g in groups:
        mt, medt, fpt = wstats(g["toward"].to_numpy(), g["w"].to_numpy())
        mk, medk, _ = wstats(g["toward_km"].to_numpy(), g["w"].to_numpy())
        lo, hi = boot_mean(g, "toward") if g["t0"].nunique() > 20 else (np.nan, np.nan)
        rows.append(dict(population=name, n=len(g), mean_toward_deg=round(mt, 4), ci95=f"{lo:+.4f} to {hi:+.4f}",
                         median_toward_deg=round(medt, 4), share_toward=round(fpt, 3), mean_toward_km=round(mk, 2), median_toward_km=round(medk, 2)))
    T = pd.DataFrame(rows)
    lines += ["", "| " + " | ".join(T.columns) + " |", "|" + "---|" * len(T.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in T.itertuples(index=False)]
    (ROOT / "SIGNED_DEVIATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
