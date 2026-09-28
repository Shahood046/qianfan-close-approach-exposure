"""Shared helpers for the Qianfan pilot: loading element sets, SGP4 setup, time and frames."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sgp4 import omm
from sgp4.api import Satrec

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

MU = 398600.8   # km^3/s^2, WGS-72 (the constants SGP4 itself uses)
RE = 6378.135   # km, WGS-72 equatorial radius

NUMERIC = ["MEAN_MOTION", "ECCENTRICITY", "INCLINATION", "RA_OF_ASC_NODE",
           "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT"]

# Space-Track and CelesTrak both serve OMM-style JSON with these keys.
GP_PREDICATES = ["NORAD_CAT_ID", "OBJECT_NAME", "OBJECT_ID", "OBJECT_TYPE", "EPOCH",
                 *NUMERIC, "CREATION_DATE", "GP_ID"]


def read_json_records(paths) -> list[dict]:
    recs = []
    for p in paths:
        p = Path(p)
        opener = gzip.open if p.suffix == ".gz" else open
        with opener(p, "rt", encoding="utf-8") as f:
            recs.extend(json.load(f))
    return recs


def load_gp(paths) -> pd.DataFrame:
    """Load GP/OMM element sets into a clean frame, one row per (object, epoch)."""
    df = pd.DataFrame(read_json_records(paths))
    if df.empty:
        raise SystemExit(f"No element sets found in {list(map(str, paths))}")
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c])
    df["NORAD_CAT_ID"] = df["NORAD_CAT_ID"].astype(int)
    df["EPOCH"] = pd.to_datetime(df["EPOCH"], utc=True, format="ISO8601")
    # Space-Track can hold re-issued element sets for the same epoch; keep the newest issue.
    if "CREATION_DATE" in df:
        df = df.sort_values("CREATION_DATE", na_position="first")
    df = (df.drop_duplicates(["NORAD_CAT_ID", "EPOCH"], keep="last")
            .sort_values(["NORAD_CAT_ID", "EPOCH"]).reset_index(drop=True))
    n = df["MEAN_MOTION"] * 2 * np.pi / 86400.0  # rad/s
    df["A_KM"] = (MU / n**2) ** (1 / 3)
    df["ALT_KM"] = df["A_KM"] - RE
    df["PERIGEE_KM"] = df["A_KM"] * (1 - df["ECCENTRICITY"]) - RE
    df["APOGEE_KM"] = df["A_KM"] * (1 + df["ECCENTRICITY"]) - RE
    df["LAUNCH"] = df["OBJECT_ID"].str[:8]
    return df


def satrec(row) -> Satrec:
    """Build an SGP4 satellite from one element-set row (mean elements, TEME frame)."""
    fields = {k: row[k] for k in NUMERIC}
    fields.update(
        NORAD_CAT_ID=int(row["NORAD_CAT_ID"]),
        OBJECT_ID=str(row["OBJECT_ID"]),
        EPOCH=row["EPOCH"].tz_convert(None).strftime("%Y-%m-%dT%H:%M:%S.%f"),
        CLASSIFICATION_TYPE="U", EPHEMERIS_TYPE=0, ELEMENT_SET_NO=999, REV_AT_EPOCH=0,
    )
    sat = Satrec()
    omm.initialize(sat, fields)
    return sat


def to_jd(times) -> tuple[np.ndarray, np.ndarray]:
    """UTC timestamps -> (jd, fr) split as SGP4 expects, without losing sub-ms precision."""
    ns = pd.DatetimeIndex(np.atleast_1d(times)).as_unit("ns").asi8
    day_ns = 86_400_000_000_000
    day = ns // day_ns
    return day + 2440587.5, (ns - day * day_ns) / day_ns


def rtn(dr: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Project difference vectors dr onto the radial / transverse (in-track) / normal frame of (r, v)."""
    R = r / np.linalg.norm(r, axis=-1, keepdims=True)
    h = np.cross(r, v)
    N = h / np.linalg.norm(h, axis=-1, keepdims=True)
    T = np.cross(N, R)
    return np.stack([(dr * R).sum(-1), (dr * T).sum(-1), (dr * N).sum(-1)], axis=-1)


def latitude_deg(r: np.ndarray) -> np.ndarray:
    """Geocentric latitude from TEME position (latitude does not depend on Earth rotation)."""
    return np.degrees(np.arcsin(r[..., 2] / np.linalg.norm(r, axis=-1)))


def load_states(tag: str = "") -> pd.DataFrame | None:
    """Per-element-set state labels written by 02_states.py (states{tag}.csv), if they exist."""
    p = DATA / f"states{tag}.csv"
    if not p.exists():
        return None
    st = pd.read_csv(p)
    st["EPOCH"] = pd.to_datetime(st["EPOCH"], utc=True, format="ISO8601")
    return st
