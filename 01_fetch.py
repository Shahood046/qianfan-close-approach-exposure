"""Download Qianfan element-set history (and optional background / CZ-6A debris) from Space-Track.

Credentials come from the environment, never the command line:
    SPACETRACK_USER, SPACETRACK_PASS

Space-Track asks users to stay under 30 requests/minute and 300/hour and to fetch gp_history
once and cache it. This script sleeps between requests and skips files that already exist.
Their user agreement forbids redistributing the raw data: keep data/ out of any public repo.

Usage:
    py 01_fetch.py                       # Qianfan satcat + full gp_history
    py 01_fetch.py --background          # + current catalog snapshot, 600-1300 km band
    py 01_fetch.py --debris              # + gp_history of the 2024-140 (CZ-6A) fragments
    py 01_fetch.py --extra-launch 2026-211 --extra-launch 2026-210
                                         # add launches whose payloads are still named "OBJECT x"
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import Counter

import requests

from common import DATA, GP_PREDICATES

BASE = "https://www.space-track.org"
PAUSE_S = 3.0      # ~20 requests/minute, under the published limit
BATCH = 10         # objects per gp_history request (keeps responses a manageable size)


def load_dotenv():
    """Read KEY=VALUE lines from .env next to this script without overriding the environment."""
    path = DATA.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip("'\""))


class SpaceTrack:
    def __init__(self):
        load_dotenv()
        user, pw = os.environ.get("SPACETRACK_USER"), os.environ.get("SPACETRACK_PASS")
        if not user or not pw:
            raise SystemExit("Set SPACETRACK_USER and SPACETRACK_PASS environment variables first.")
        self.s = requests.Session()
        r = self.s.post(f"{BASE}/ajaxauth/login", data={"identity": user, "password": pw}, timeout=60)
        if r.status_code != 200 or "Failed" in r.text:
            raise SystemExit(f"Space-Track login failed (HTTP {r.status_code}).")
        self.n = 0

    def query(self, path: str) -> list[dict]:
        if self.n:
            time.sleep(PAUSE_S)
        self.n += 1
        url = f"{BASE}/basicspacedata/query/{path}"
        for attempt in range(3):
            r = self.s.get(url, timeout=300)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(60 * (attempt + 1))
                continue
            break
        raise SystemExit(f"Query failed (HTTP {r.status_code}): {path}\n{r.text[:300]}")


def save(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def load(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def fetch_satcat(st: SpaceTrack, extra_launches: list[str]) -> list[dict]:
    path = DATA / "satcat_qianfan.json.gz"
    if path.exists():
        return load(path)
    named = st.query("class/satcat/SATNAME/~~QIANFAN/orderby/NORAD_CAT_ID asc/format/json")
    launches = sorted({o["INTLDES"][:8] for o in named} | set(extra_launches))
    # Pull every payload from those launches: freshly launched objects are often still "OBJECT A".
    payloads = []
    for lid in launches:
        payloads += st.query(f"class/satcat/INTLDES/~~{lid}/OBJECT_TYPE/PAYLOAD/format/json")
    keep, report = [], []
    for lid in launches:
        objs = [o for o in payloads if o["INTLDES"].startswith(lid)]
        n_qf = sum("QIANFAN" in o["SATNAME"].upper() for o in objs)
        # A launch counts as a Qianfan launch if most payloads carry the name (or you listed it).
        whole = lid in extra_launches or n_qf >= 0.5 * len(objs)
        chosen = objs if whole else [o for o in objs if "QIANFAN" in o["SATNAME"].upper()]
        keep += chosen
        report.append((lid, len(objs), n_qf, len(chosen)))
    print("launch     payloads  named-QIANFAN  kept")
    for lid, n, q, k in report:
        print(f"{lid}   {n:8d}  {q:13d}  {k:4d}")
    save(keep, path)
    return keep


def fetch_history(st: SpaceTrack, ids: list[int], subdir: str):
    preds = ",".join(GP_PREDICATES)
    out = DATA / subdir
    for b in range(0, len(ids), BATCH):
        chunk = ids[b:b + BATCH]
        path = out / f"{chunk[0]}_{chunk[-1]}.json.gz"
        if path.exists():
            continue
        q = (f"class/gp_history/NORAD_CAT_ID/{','.join(map(str, chunk))}"
             f"/orderby/NORAD_CAT_ID asc,EPOCH asc/predicates/{preds}/format/json")
        recs = st.query(q)
        save(recs, path)
        print(f"  {subdir}: objects {chunk[0]}..{chunk[-1]} -> {len(recs)} element sets")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--background", action="store_true", help="current catalog snapshot, 600-1300 km")
    ap.add_argument("--debris", action="store_true", help="gp_history of CZ-6A 2024-140 fragments")
    ap.add_argument("--extra-launch", action="append", default=[], metavar="YYYY-NNN")
    args = ap.parse_args()

    st = SpaceTrack()
    sats = fetch_satcat(st, args.extra_launch)
    ids = sorted(int(o["NORAD_CAT_ID"]) for o in sats)
    print(f"{len(ids)} Qianfan payloads in {len(Counter(o['INTLDES'][:8] for o in sats))} launches")
    fetch_history(st, ids, "gp_history_qianfan")

    if args.debris:
        path = DATA / "satcat_cz6a_2024140.json.gz"
        if not path.exists():
            save(st.query("class/satcat/INTLDES/~~2024-140/OBJECT_TYPE/<>PAYLOAD/format/json"), path)
        frag = sorted(int(o["NORAD_CAT_ID"]) for o in load(path))
        print(f"{len(frag)} non-payload objects from 2024-140")
        fetch_history(st, frag, "gp_history_cz6a")

    if args.background:
        path = DATA / "background_snapshot.json.gz"
        if not path.exists():
            preds = ",".join(GP_PREDICATES)
            recs = st.query("class/gp/DECAY_DATE/null-val/EPOCH/>now-10/PERIAPSIS/<1300/APOAPSIS/>600"
                            f"/orderby/NORAD_CAT_ID asc/predicates/{preds}/format/json")
            save(recs, path)
            print(f"background snapshot: {len(recs)} objects")

    print(f"done ({st.n} requests)")


if __name__ == "__main__":
    main()
