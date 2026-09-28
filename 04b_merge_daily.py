"""Merge the two daily screens into the record used by 11_per_angle.py and 14_locked_claims.py.

    out/encounter_pairs_dailyB.csv   2025-01-01 .. 2026-03-31   (py 04_screen.py --every-days 1 --start 2025-01-01 --end 2026-03-31 --kinds QQ --tag _dailyB)
    out/encounter_pairs_dailyA.csv   2026-04-01 .. 2026-09-22   (py 04_screen.py --every-days 1 --start 2026-04-01 --end 2026-09-22 --kinds QQ --tag _dailyA)
    -> out/encounter_pairs_daily.csv

The merge is a concatenation, sorted on (t0, NORAD_CAT_ID_1, NORAD_CAT_ID_2). When this step was first done by hand for the analysis, the
merged file held the same 14,150 rows as a set: after sorting on that key every column is identical, except that four floating-point columns
differ by round-off only (largest difference 3.6e-15). The original row order was not recorded and is not preserved.

Row order does not matter downstream (static audit and a check of the order-exposed steps, 2026-09-28):
  * 11_per_angle.py takes the window list from sorted(t0), reduces the screen with groupby(t0, a, b).min(), and builds every pair-window from the
    element sets themselves; the screen only supplies d_min through a key merge, so the pair-window table does not depend on the screen's row order;
    relabel() sorts before merge_asof and merges on keys;
  * 14_locked_claims.py merges the screen on (t0, a, b) and takes medians, which are order-independent;
  * the pilot consumers (06_figures.py, 09_model_checks.py part B) only use groupby, nunique and pivot_table.
Nothing downstream uses positional access, shift, diff, rolling, head or tail on the screen. Use --check to compare a fresh merge with an existing file.

Usage:  py 04b_merge_daily.py            (writes out/encounter_pairs_daily.csv; refuses to overwrite an existing file unless --force)
        py 04b_merge_daily.py --check    (compares a fresh merge with the existing file, writes nothing)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import OUT

KEY = ["t0", "NORAD_CAT_ID_1", "NORAD_CAT_ID_2"]


def merged() -> pd.DataFrame:
    b = pd.read_csv(OUT / "encounter_pairs_dailyB.csv")
    a = pd.read_csv(OUT / "encounter_pairs_dailyA.csv")
    return pd.concat([b, a], ignore_index=True).sort_values(KEY, kind="stable").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    m = merged()
    target = OUT / "encounter_pairs_daily.csv"
    if args.check:
        old = pd.read_csv(target).sort_values(KEY, kind="stable").reset_index(drop=True)
        same = len(m) == len(old) and list(m.columns) == list(old.columns)
        bad = []
        if same:
            for c in m.columns:
                ok = (np.allclose(m[c].to_numpy(dtype=float), old[c].to_numpy(dtype=float), rtol=1e-9, atol=0, equal_nan=True)
                      if m[c].dtype.kind in "fi" else (m[c].astype(str) == old[c].astype(str)).all())
                if not ok:
                    bad.append(c)
        print(f"fresh merge {len(m):,} rows, existing file {len(old):,} rows; same content after sorting on {KEY} "
              f"(numeric columns to 1e-9 relative): {same and not bad}" + (f"; differing columns: {bad}" if bad else ""))
        return
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; use --check to compare or --force to overwrite")
    m.to_csv(target, index=False)
    print(f"wrote {target} ({len(m):,} rows)")


if __name__ == "__main__":
    main()
