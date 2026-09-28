"""Test of the 2026 event-rate rise by plane-pair class: composition or within-class change?

Classes of primary pair-windows (on-grid, slowly drifting, standard angle; data/pair_windows_daily.pkl.gz):
    even_coinciding   even plane step, occupied slots of the two planes coincide at the crossing in that window
    even_partial      even plane step, only a minority (10-50%) of either plane's satellites has a coinciding partner
    even_interleaved  even plane step, occupied slots interleaved (satellites pass ~10 deg minus the grid offset apart)
    odd               odd plane step
    sparse            a pair involving the sparse plane of launch 2024-185 (grid not reliable)

Slot coincidence is decided per window from the actual pair-level nominal margins (no plane-level offsets): for each
on-grid, slowly drifting satellite, its closest partner in the other plane is "coinciding" if the pair's nominal margin
is below the half-slot distance r * 5 deg * cos(gamma/2) (the midpoint between a coinciding slot at offset delta and an
interleaved one at 10 deg - delta). The plane pair coincides if most satellites of either plane have such a partner
(a plane filled by two launches gives only half of its own satellites a coinciding partner).

Planes are identified by launch (each launch populated exactly one plane; the launch-to-plane map is read from the
final window). Writes RISE_TEST.md.

Usage:  py 18_rise_test.py
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from common import DATA, RE, ROOT, load_gp, load_states

_spec = importlib.util.spec_from_file_location("geo", ROOT / "17_geometry.py")
geo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(geo)

A_KM = RE + 1068.5
PERIODS = [("2025-03-13", "2026-04-01", "P0 2025-03-13..2026-03-31"), ("2026-04-01", "2026-06-01", "P1 2026-04-01..05-31"),
           ("2026-06-01", "2026-09-01", "P2 2026-06-01..08-31"), ("2026-09-01", "2026-09-23", "P3 2026-09-01..09-22")]
HALVES = [("2025-03-13", "2026-04-01", "early"), ("2026-04-01", "2026-09-23", "recent")]
CLASSES = ["even_coinciding", "even_partial", "even_interleaved", "odd", "sparse"]


def md_table(df: pd.DataFrame) -> str:
    return geo.md_table(df)


def plane_map() -> dict:
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    el, T = geo.snapshot(df, load_states(), geo.FINAL)
    lab = dict(zip(T["plane"], T["label"]))
    m = el.groupby("LAUNCH")["plane"].agg(lambda p: p.mode().iloc[0]).map(lab)
    return m.to_dict()


def classify(P: pd.DataFrame, l2p: dict) -> pd.DataFrame:
    P = P.assign(pa=P["a"].str[:8].map(l2p), pb=P["b"].str[:8].map(l2p))
    P["pp"] = ["-".join(sorted([x, y], key=lambda s: int(s[1:]))) for x, y in zip(P["pa"], P["pb"])]
    P["half_slot_km"] = A_KM * np.radians(5.0) * np.cos(np.radians(P["gamma"]) / 2)
    good = P[P["good"]]
    rows = []
    for (t0, pp), g in good.groupby(["t0", "pp"], sort=False):
        long = pd.concat([g[["a", "pa", "nominal", "half_slot_km"]].rename(columns={"a": "sat", "pa": "plane"}),
                          g[["b", "pb", "nominal", "half_slot_km"]].rename(columns={"b": "sat", "pb": "plane"})])
        per = long.groupby(["plane", "sat"]).agg(nom=("nominal", "min"), h=("half_slot_km", "first"))
        hit = (per["nom"] < per["h"]).groupby(level=0).mean()
        rows.append((t0, pp, float(hit.max())))
    C = pd.DataFrame(rows, columns=["t0", "pp", "coincide_frac"])
    P = P.merge(C, on=["t0", "pp"], how="left")
    sparse = (P["pa"] == "P3") | (P["pb"] == "P3")
    even = P["k"] % 2 == 0
    cf = P["coincide_frac"].fillna(0)
    P["cls"] = np.select([sparse, ~even, even & (cf > 0.5), even & (cf > 0.1)],
                         ["sparse", "odd", "even_coinciding", "even_partial"], "even_interleaved")
    return P


def table(G: pd.DataFrame, periods) -> pd.DataFrame:
    rows = []
    for lo, hi, name in periods:
        X = G[(G["t0"] >= pd.Timestamp(lo, tz="UTC")) & (G["t0"] < pd.Timestamp(hi, tz="UTC"))]
        tot_n, tot_e = len(X), int(X["obs10"].sum())
        rows.append(dict(period=name, cls="ALL", pair_windows=tot_n, events=tot_e, rate=1000 * tot_e / tot_n,
                         share_pw=1.0, share_ev=1.0, med_nominal=X["nominal"].median(), plane_pairs=X["pp"].nunique()))
        for c in CLASSES:
            Y = X[X["cls"] == c]
            e = int(Y["obs10"].sum())
            rows.append(dict(period=name, cls=c, pair_windows=len(Y), events=e, rate=1000 * e / len(Y) if len(Y) else np.nan,
                             share_pw=len(Y) / tot_n, share_ev=e / tot_e if tot_e else np.nan,
                             med_nominal=Y["nominal"].median() if len(Y) else np.nan, plane_pairs=Y["pp"].nunique()))
    T = pd.DataFrame(rows)
    for c, d in (("rate", 3), ("share_pw", 3), ("share_ev", 3), ("med_nominal", 0)):
        T[c] = T[c].round(d)
    return T


def decompose(T: pd.DataFrame, a: str, b: str) -> dict:
    """Kitagawa decomposition of the total rate change a -> b into composition and within-class parts."""
    A = T[(T["period"] == a) & (T["cls"] != "ALL")].set_index("cls")
    B = T[(T["period"] == b) & (T["cls"] != "ALL")].set_index("cls")
    ra, rb_ = A["rate"].fillna(0), B["rate"].fillna(0)
    sa, sb = A["share_pw"], B["share_pw"]
    comp = ((sb - sa) * (ra + rb_) / 2).sum()
    within = ((rb_ - ra) * (sa + sb) / 2).sum()
    per_cls_within = ((rb_ - ra) * (sa + sb) / 2).round(3).to_dict()
    per_cls_comp = ((sb - sa) * (ra + rb_) / 2).round(3).to_dict()
    total = float(T[(T["period"] == b) & (T["cls"] == "ALL")]["rate"].iloc[0] - T[(T["period"] == a) & (T["cls"] == "ALL")]["rate"].iloc[0])
    return dict(total=total, composition=comp, within=within, comp_by_cls=per_cls_comp, within_by_cls=per_cls_within)


def margin_bins(G: pd.DataFrame, periods) -> pd.DataFrame:
    bins = [0, 10, 20, 40, 60, 100, np.inf]
    G = G.assign(mbin=pd.cut(G["nominal"], bins, right=False))
    rows = []
    for lo, hi, name in periods:
        X = G[(G["t0"] >= pd.Timestamp(lo, tz="UTC")) & (G["t0"] < pd.Timestamp(hi, tz="UTC")) & (G["cls"] == "even_coinciding")]
        for b, Y in X.groupby("mbin", observed=True):
            rows.append(dict(period=name, nominal_bin=str(b), pair_windows=len(Y), events=int(Y["obs10"].sum()),
                             rate=round(1000 * Y["obs10"].mean(), 1)))
    return pd.DataFrame(rows)


def by_plane_pair(G: pd.DataFrame, periods) -> pd.DataFrame:
    rows = []
    X = G[G["cls"] == "even_coinciding"]
    for lo, hi, name in periods:
        Y = X[(X["t0"] >= pd.Timestamp(lo, tz="UTC")) & (X["t0"] < pd.Timestamp(hi, tz="UTC"))]
        for pp, Z in Y.groupby("pp"):
            rows.append(dict(period=name, plane_pair=pp, k=int(Z["k"].mode().iloc[0]), days=Z["t0"].nunique(), pair_windows=len(Z),
                             events=int(Z["obs10"].sum()), rate=round(1000 * Z["obs10"].mean(), 2),
                             min_nominal_median=round(float(Z.groupby("t0")["nominal"].min().median()), 1)))
    return pd.DataFrame(rows)


def boot_decomp(G: pd.DataFrame, a: tuple, b: tuple, nb: int = 500, block: int = 7, seed: int = 3) -> str:
    """7-day moving-block bootstrap (within each period) of the composition / within-class split between periods a and b."""
    W = G.groupby(["t0", "cls"])["obs10"].agg(["size", "sum"]).unstack(fill_value=0)
    rng = np.random.default_rng(seed)

    def pick(lo, hi):
        days = W.index[(W.index >= pd.Timestamp(lo, tz="UTC")) & (W.index < pd.Timestamp(hi, tz="UTC"))]
        nblk = int(np.ceil(len(days) / block))
        starts = rng.integers(0, max(len(days) - block, 1), nblk)
        return W.loc[[days[s + j] for s in starts for j in range(block) if s + j < len(days)]].sum()

    out = []
    for _ in range(nb):
        rows = []
        for lo, hi, name in (a, b):
            s = pick(lo, hi)
            n, e = s["size"], s["sum"]
            for c in CLASSES:
                rows.append(dict(period=name, cls=c, share_pw=n.get(c, 0) / n.sum(), rate=1000 * e.get(c, 0) / n.get(c, 1) if n.get(c, 0) else np.nan))
            rows.append(dict(period=name, cls="ALL", share_pw=1.0, rate=1000 * e.sum() / n.sum()))
        d = decompose(pd.DataFrame(rows), a[2], b[2])
        out.append((d["total"], d["composition"], d["within"]))
    o = np.array(out)
    q = lambda x: f"{np.percentile(x, 2.5):+.3f} to {np.percentile(x, 97.5):+.3f}"
    return f"total {q(o[:, 0])}; composition {q(o[:, 1])}; within-class {q(o[:, 2])}"


def main():
    l2p = plane_map()
    P = pd.read_pickle(DATA / "pair_windows_daily.pkl.gz")
    P["t0"] = pd.to_datetime(P["t0"], utc=True)
    P = classify(P, l2p)
    G = P[P["good"] & P["std_angle"]]
    lines = ["# 2026 rise test by plane-pair class (auto-generated by 18_rise_test.py)", "",
             f"Launch -> plane: {l2p}", f"Primary pair-windows: {len(G):,}; events {int(G['obs10'].sum())}",
             f"Class counts (primary): {G['cls'].value_counts().to_dict()}", ""]
    for title, per in (("Early vs recent", HALVES), ("Four periods", PERIODS)):
        T = table(G, per)
        lines += [f"## {title}", "", md_table(T), ""]
        names = [p[2] for p in per]
        for a, b in zip(names[:-1], names[1:]):
            d = decompose(T, a, b)
            lines.append(f"- {a} -> {b}: total rate change {d['total']:+.3f} per 1000 = composition {d['composition']:+.3f} "
                         f"+ within-class {d['within']:+.3f}; within by class {d['within_by_cls']}; composition by class {d['comp_by_cls']}")
        if title == "Four periods":
            d = decompose(T, names[0], names[-1])
            lines.append(f"- {names[0]} -> {names[-1]}: total {d['total']:+.3f} = composition {d['composition']:+.3f} + within {d['within']:+.3f}")
        lines.append("")
    lines += ["## Decomposition uncertainty (7-day moving-block bootstrap within each period, 500 draws, 95% intervals)", "",
              f"- early -> recent: {boot_decomp(G, HALVES[0], HALVES[1])}",
              f"- {PERIODS[0][2]} -> {PERIODS[1][2]}: {boot_decomp(G, PERIODS[0], PERIODS[1])}",
              f"- {PERIODS[1][2]} -> {PERIODS[2][2]}: {boot_decomp(G, PERIODS[1], PERIODS[2])}",
              f"- {PERIODS[0][2]} -> {PERIODS[3][2]}: {boot_decomp(G, PERIODS[0], PERIODS[3])}", ""]
    lines += ["## Even-coinciding class: rate by nominal-margin bin and period", "", md_table(margin_bins(G, PERIODS)), ""]
    lines += ["## Even-coinciding class by plane pair and period", "", md_table(by_plane_pair(G, PERIODS)), ""]
    (ROOT / "RISE_TEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
