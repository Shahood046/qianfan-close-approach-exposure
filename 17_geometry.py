"""Reconstructed as-operated geometry for manuscript Section 4, written to GEOMETRY_CLAIMS.md.

Per plane: population, launches, RAAN, inclination, altitude, eccentricity vector, slot-grid phase and coherence.
Per plane pair: crossing angle, plane step k, grid-to-grid phase offset at the two plane crossings (mod the 10 deg slot
spacing), radial offset at the crossings, and the resulting plane-level nominal margin. Stability over time from
snapshots every 30 days. Same element-set selection and estimators as 08_robustness.py / 11_per_angle.py.

Usage:  py 17_geometry.py
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd

from common import DATA, OUT, RE, ROOT, load_gp, load_states

_spec = importlib.util.spec_from_file_location("rb", ROOT / "08_robustness.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
cm, pl = rb.cm, rb.pl

S = 10.0
SPACING = 20.5
MIN_PLANE = 5          # planes with fewer satellites are "sparse": their slot grid is not defined reliably
FINAL = pd.Timestamp("2026-09-22 12:00", tz="UTC")


def md_table(df: pd.DataFrame) -> str:
    fmt = lambda v: "" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v))
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def plane_table(el: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, g in el.groupby("plane"):
        ex = (g["ECCENTRICITY"] * np.cos(np.radians(g["ARG_OF_PERICENTER"]))).mean()
        ey = (g["ECCENTRICITY"] * np.sin(np.radians(g["ARG_OF_PERICENTER"]))).mean()
        raan = np.unwrap(np.radians(g["raan"].to_numpy())) * 180 / np.pi
        rows.append(dict(plane=p, n=len(g), launches=",".join(sorted(set(g["LAUNCH"]))),
                         raan=pl.circmean(g["raan"].to_numpy(), 360), raan_spread=np.ptp(raan),
                         inc=g["inc"].mean(), inc_spread=np.ptp(g["inc"]),
                         a=g["A_KM"].median(), alt=g["A_KM"].median() - RE, a_spread=np.ptp(g["A_KM"]),
                         e=float(np.hypot(ex, ey)), argp=float(np.degrees(np.arctan2(ey, ex)) % 360),
                         phase=pl.circmean(g["um"].to_numpy() % S, S),
                         R10=abs(np.exp(2j * np.pi * g["um"].to_numpy() / S).mean()),
                         within1=(g["slot_err_deg"].abs() <= 1).mean()))
    T = pd.DataFrame(rows).sort_values("raan").reset_index(drop=True)
    T["label"] = [f"P{i + 1}" for i in range(len(T))]
    T["sparse"] = T["n"] < MIN_PLANE
    return T


def lattice_spacing(T: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Cyclic RAAN gaps between populated planes, largest gap (outside the star) dropped, each divided by its
    number of lattice steps. Returns per-step spacings and the angular span of the populated planes."""
    r = np.sort(T.loc[~T["sparse"], "raan"].to_numpy())
    gaps = np.diff(np.r_[r, r[0] + 360])
    big = gaps.argmax()
    keep = np.delete(gaps, big)
    steps = np.maximum(np.round(keep / SPACING), 1)
    return np.repeat(keep / steps, steps.astype(int)), 360 - gaps[big]


def pair_table(T: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(T)):
        for j in range(i + 1, len(T)):
            a, b = T.iloc[i], T.iloc[j]
            h1, h2 = cm.unit_h(np.array([a["raan"]]), np.array([a["inc"]])), cm.unit_h(np.array([b["raan"]]), np.array([b["inc"]]))
            line = np.cross(h1, h2)
            line /= np.linalg.norm(line, axis=1, keepdims=True)
            gamma = float(np.degrees(np.arccos(np.clip((h1 * h2).sum(), -1, 1))))
            best = None
            for sgn in (1, -1):
                p = sgn * line
                us1 = cm.u_of_direction(p, np.array([a["raan"]]), np.array([a["inc"]]))[0]
                us2 = cm.u_of_direction(p, np.array([b["raan"]]), np.array([b["inc"]]))[0]
                off = rb.wrap_s((a["phase"] - cm.mean_phase(us1, a["e"], a["argp"])) - (b["phase"] - cm.mean_phase(us2, b["e"], b["argp"])))
                r1 = a["a"] * (1 - a["e"] * np.cos(np.radians(us1 - a["argp"])))
                r2 = b["a"] * (1 - b["e"] * np.cos(np.radians(us2 - b["argp"])))
                along = 0.5 * (r1 + r2) * np.radians(off) * np.cos(np.radians(gamma) / 2)
                d = float(np.hypot(along, r1 - r2))
                lat = float(np.degrees(np.arcsin(p[0, 2])))
                if best is None or d < best["margin_km"]:
                    best = dict(offset_deg=float(off), along_km=float(along), radial_km=float(r1 - r2), margin_km=d, lat=lat)
            k = int(round(gamma / SPACING))
            rows.append(dict(pair=f"{a['label']}-{b['label']}", sparse=bool(a["sparse"] or b["sparse"]), gamma=gamma, k=k,
                             std=abs(gamma - k * SPACING) < 2, **best))
    return pd.DataFrame(rows)


def add_pair_level(PT: pd.DataFrame, T: pd.DataFrame, el: pd.DataFrame, Pw: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Join the smallest pair-level nominal margin (actual satellites, slot errors removed; 11_per_angle) for each plane
    pair at the window whose centre is t, and flag whether the occupied slots of the two planes coincide at the crossing.

    Single-launch planes occupy alternate slots of the 10 deg grid, so two planes whose grids are aligned (mod 10 deg) can
    still have their satellites interleaved, passing (10 - |offset|) deg apart instead of |offset|."""
    lab = dict(zip(T["plane"], T["label"]))
    o2p = {o: lab[p] for o, p in zip(el["OBJECT_ID"], el["plane"])}
    F = Pw[(Pw["t0"] == t - pd.Timedelta(hours=12)) & Pw["good"]]          # on-grid, slowly drifting pairs only
    F = F.assign(pa=F["a"].map(o2p), pb=F["b"].map(o2p)).dropna(subset=["pa", "pb"])
    F = F.assign(pair=["-".join(sorted([x, y], key=lambda s: int(s[1:]))) for x, y in zip(F["pa"], F["pb"])])
    X = PT.copy()
    r = T["a"].median()
    X["shifted_km"] = np.hypot(r * np.radians(10 - X["offset_deg"].abs()) * np.cos(np.radians(X["gamma"]) / 2), X["radial_km"])
    mins, frac = {}, {}
    for pair, g in F.groupby("pair"):
        row = X[X["pair"] == pair]
        if row.empty:
            continue
        m0, m1 = float(row["margin_km"].iloc[0]), float(row["shifted_km"].iloc[0])
        # each on-grid satellite's closest partner in the other plane: coinciding slot (~m0) or interleaved (~m1)?
        # A plane occupying both slot parities (two launches) gives only half of its satellites a coinciding partner, so
        # the occupied slot sets overlap if most satellites of EITHER plane have one.
        long = pd.concat([g[["a", "nominal"]].rename(columns={"a": "sat"}), g[["b", "nominal"]].rename(columns={"b": "sat"})])
        per_sat = long.groupby("sat")["nominal"].min()          # a satellite can appear in either column
        hit = ((per_sat - m0).abs() < (per_sat - m1).abs()).groupby(per_sat.index.map(o2p)).mean()
        frac[pair] = float(hit.max())
        mins[pair] = float(g["nominal"].min())
    X["pair_min_nominal_km"] = X["pair"].map(mins)
    X["coincide_frac"] = X["pair"].map(frac)
    X["slots_coincide"] = X["coincide_frac"] > 0.5
    return X


def rb_share(G: pd.DataFrame, frac: float = 0.01) -> float:
    """Share of events in the smallest `frac` of pair-windows by nominal margin (same rule as 14_locked_claims.share_at)."""
    o = np.argsort(G["nominal"].to_numpy(), kind="stable")
    cum = np.cumsum(G["obs10"].to_numpy()[o]) / max(G["obs10"].sum(), 1)
    return float(cum[max(int(frac * len(G)) - 1, 0)])


def snapshot(df, states, t):
    el = rb.slots(rb.get_el(df, states, t), rb.ESTIMATORS["circular mean"])
    return el, plane_table(el)


def main():
    df = load_gp(sorted((DATA / "gp_history_qianfan").glob("*.json.gz")))
    states = load_states()
    lines = ["# Geometry claims for Section 4 (auto-generated by 17_geometry.py)", ""]
    add = lines.append

    el, T = snapshot(df, states, FINAL)
    # tables kept for the figures (Figs. 4-6: coherence scan, plane geometry, plane pairs, snapshot offsets); no per-satellite table is written
    lab_ = dict(zip(T["plane"], T["label"]))
    T.to_csv(OUT / "final_planes.csv", index=False)
    grid_ = np.round(np.arange(4, 25.001, 0.05), 2)
    pd.DataFrame({"s_deg": grid_, "R": [rb.coherence(el, s_) for s_ in grid_]}).to_csv(OUT / "coherence_scan.csv", index=False)
    add(f"## Planes at {FINAL:%Y-%m-%d %H:%M} UTC ({len(el)} operational satellites)")
    add("")
    show = T[["label", "n", "launches", "raan", "raan_spread", "inc", "inc_spread", "alt", "a_spread", "e", "argp", "phase", "R10", "within1"]].copy()
    for c, d in (("raan", 2), ("raan_spread", 2), ("inc", 3), ("inc_spread", 3), ("alt", 1), ("a_spread", 2), ("argp", 0), ("phase", 2), ("R10", 3), ("within1", 3)):
        show[c] = show[c].round(d)
    show["e"] = show["e"].map(lambda v: f"{v:.5f}")
    add(md_table(show))
    add("")
    sp, span = lattice_spacing(T)
    add(f"- Populated planes (>= {MIN_PLANE} satellites): {int((~T['sparse']).sum())}; sparse: "
        + (", ".join(f"{r.label} ({r.n} satellites, {r.launches})" for r in T[T['sparse']].itertuples()) or "none"))
    add(f"- RAAN spacing per lattice step between populated planes: " + ", ".join(f"{v:.2f}" for v in sp)
        + f" deg (median {np.median(sp):.2f}, range {sp.min():.2f}-{sp.max():.2f}); populated planes span {span:.2f} deg of RAAN")
    allgap = np.diff(np.sort(T["raan"].to_numpy()))
    add(f"- Largest within-plane RAAN spread {T['raan_spread'].max():.2f} deg; smallest gap between adjacent planes (any) {allgap.min():.2f} deg "
        f"-> any clustering gap between these values gives the same planes")
    g185 = el[el["LAUNCH"] == "2024-185"]
    if len(g185):
        p185 = T.set_index("plane").loc[g185["plane"].iloc[0]]
        add(f"- Launch 2024-185: {len(g185)} operational satellites in plane {p185['label']} (RAAN {g185['raan'].min():.2f}-{g185['raan'].max():.2f} deg; "
            f"plane mean {p185['raan']:.2f}); crossing angles of pairs involving them are offset from the 20.5 deg lattice")
    add(f"- Mean-phase equation-of-centre amplitude 2e: " + ", ".join(f"{l} {np.degrees(2 * e):.3f} deg" for l, e in zip(T["label"], T["e"])))
    add(f"- Radial amplitude a*e: " + ", ".join(f"{l} {a * e:.1f} km" for l, a, e in zip(T["label"], T["a"], T["e"])))
    add("")

    Pw = pd.read_pickle(DATA / "pair_windows_daily.pkl.gz")
    Pw["t0"] = pd.to_datetime(Pw["t0"], utc=True)
    add("## Slot occupancy (10 deg grid = 36 slots per plane; slot index parity relative to the plane's first satellite)")
    for p, g in el.groupby("plane"):
        ref = g["um"].iloc[0] - g["slot_err_deg"].iloc[0]
        idx = np.round(((g["um"] - g["slot_err_deg"] - ref) % 360) / S).astype(int) % 36
        par = {l: sorted(set(idx[g["LAUNCH"] == l] % 2)) for l in sorted(set(g["LAUNCH"]))}
        add(f"- {dict(zip(T['plane'], T['label']))[p]}: {len(g)} satellites in {idx.nunique()} distinct slots; parity by launch {par}")
    add("")

    PT = add_pair_level(pair_table(T), T, el, Pw, FINAL)
    PT.to_csv(OUT / "final_pair_table.csv", index=False)
    add("## Plane pairs: grid-to-grid offset at the closer of the two crossings (plane-level, slot errors removed)")
    add("margin_km = plane-level margin if occupied slots coincided; pair_min_nominal_km = smallest nominal margin among the actual "
        "satellite pairs of the two planes (11_per_angle, same window); slots_coincide = which of the two it matches.")
    add("")
    show = PT.copy()
    for c, d in (("gamma", 2), ("offset_deg", 2), ("along_km", 1), ("radial_km", 1), ("margin_km", 1), ("lat", 1), ("pair_min_nominal_km", 1), ("shifted_km", 1), ("coincide_frac", 2)):
        show[c] = show[c].round(d)
    add(md_table(show.sort_values(["k", "pair"])))
    add("")
    PTp = PT[PT["std"] & ~PT["sparse"]]
    add("Summary over standard-angle pairs of populated planes (pairs with a sparse plane excluded):")
    for par, X in (("even k", PTp[PTp["k"] % 2 == 0]), ("odd k", PTp[PTp["k"] % 2 == 1])):
        add(f"- {par}: grid |offset| median {X['offset_deg'].abs().median():.2f} deg (range {X['offset_deg'].abs().min():.2f}-"
            f"{X['offset_deg'].abs().max():.2f}); n = {len(X)}; occupied slots coincide in {int(X['slots_coincide'].sum())}; "
            f"smallest actual nominal margin: coinciding {X.loc[X['slots_coincide'], 'pair_min_nominal_km'].min():.0f}-"
            f"{X.loc[X['slots_coincide'], 'pair_min_nominal_km'].max():.0f} km, interleaved "
            + (f"{X.loc[~X['slots_coincide'], 'pair_min_nominal_km'].min():.0f}-{X.loc[~X['slots_coincide'], 'pair_min_nominal_km'].max():.0f} km"
               if (~X['slots_coincide']).any() else "none"))
    add("")

    add("## Stability over time (snapshots every 30 d)")
    add("")
    rows, offs = [], []
    for t in pd.date_range(pd.Timestamp("2025-04-01 12:00", tz="UTC"), FINAL, freq="30D"):
        try:
            e_t, T_t = snapshot(df, states, t)
        except ValueError:
            continue
        if len(T_t) < 2:
            continue
        if (~T_t["sparse"]).sum() < 2:
            continue
        s_t, _ = lattice_spacing(T_t)
        P_t = add_pair_level(pair_table(T_t), T_t, e_t, Pw, t)
        P_t = P_t[P_t["std"] & ~P_t["sparse"]]
        ev, od = P_t[P_t["k"] % 2 == 0], P_t[P_t["k"] % 2 == 1]
        rows.append(dict(date=f"{t:%Y-%m-%d}", n=len(e_t), planes=int((~T_t["sparse"]).sum()), sparse=int(T_t["sparse"].sum()),
                         spacing_median=round(float(np.median(s_t)), 2), spacing_min=round(float(s_t.min()), 2), spacing_max=round(float(s_t.max()), 2),
                         max_within_spread=round(float(T_t.loc[~T_t["sparse"], "raan_spread"].max()), 2), R10=round(rb.coherence(e_t, 10), 3),
                         even_abs_offset_med=round(float(ev["offset_deg"].abs().median()), 2) if len(ev) else np.nan,
                         odd_abs_offset_med=round(float(od["offset_deg"].abs().median()), 2) if len(od) else np.nan,
                         odd_abs_offset_min=round(float(od["offset_deg"].abs().min()), 2) if len(od) else np.nan,
                         even_pairs_coinciding=f"{int(ev['slots_coincide'].sum())}/{len(ev)}",
                         odd_min_actual_km=round(float(od["pair_min_nominal_km"].min()), 0) if len(od) else np.nan))
        offs.append(P_t.assign(date=t))
    add(md_table(pd.DataFrame(rows)))
    add("")
    O = pd.concat(offs)
    O.to_csv(OUT / "geometry_snapshot_offsets.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "geometry_snapshots.csv", index=False)
    add(f"- All snapshots, standard-angle plane pairs: even k |offset| median {O[O['k'] % 2 == 0]['offset_deg'].abs().median():.2f} deg, "
        f"P90 {O[O['k'] % 2 == 0]['offset_deg'].abs().quantile(.9):.2f}; odd k |offset| median {O[O['k'] % 2 == 1]['offset_deg'].abs().median():.2f}, "
        f"min {O[O['k'] % 2 == 1]['offset_deg'].abs().min():.2f} deg ({len(O)} plane-pair snapshots)")

    add("")
    add("## Contribution of the sparse 2024-185 plane to the exposure population (daily record)")
    inv = Pw["a"].str.startswith("2024-185") | Pw["b"].str.startswith("2024-185")
    G = Pw[Pw["good"] & Pw["std_angle"]]
    gi = G["a"].str.startswith("2024-185") | G["b"].str.startswith("2024-185")
    add(f"- All operational inter-plane pair-windows involving 2024-185: {int(inv.sum()):,} ({inv.mean():.1%}), events {int(Pw[inv]['obs10'].sum())} "
        f"of {int(Pw['obs10'].sum())}")
    add(f"- Primary population involving 2024-185: {int(gi.sum()):,} ({gi.mean():.1%}), events {int(G[gi]['obs10'].sum())} of {int(G['obs10'].sum())}; "
        f"odd-k events involving 2024-185: {int(G[gi & (G['k'] % 2 == 1)]['obs10'].sum())} of {int(G[G['k'] % 2 == 1]['obs10'].sum())}")
    H = G[~gi]
    e, o = 1000 * H[H["k"] % 2 == 0]["obs10"].mean(), 1000 * H[H["k"] % 2 == 1]["obs10"].mean()
    sh = rb_share(H)
    add(f"- Primary population without 2024-185: {len(H):,} pair-windows, {int(H['obs10'].sum())} events, rate {1000 * H['obs10'].mean():.3f} per 1000, "
        f"1% concentration {sh:.1%}, even/odd {e:.2f}/{o:.3f} (ratio {e / o:.0f})")

    (ROOT / "GEOMETRY_CLAIMS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
