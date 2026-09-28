# Space-Track query specification

This is the query procedure behind the dataset of Section 3.1, written from `01_fetch.py` and `common.py` and checked against the stored
data on 2026-09-28. Raw element sets are not part of the release (Space-Track User Agreement); a reader reproduces the dataset from their own
Space-Track account with `01_fetch.py`.

## Access

- Endpoint `https://www.space-track.org`, login through `/ajaxauth/login`.
- Credentials are read from the environment variables `SPACETRACK_USER` and `SPACETRACK_PASS` (or a `.env` file next to the scripts, which is
  never released). They are never passed on the command line.
- Request pacing: 3 s between requests (about 20 per minute, under Space-Track's published limits); existing cached files are not fetched again.

## Step 1: payload list (`class/satcat`)

1. `class/satcat/SATNAME/~~QIANFAN/orderby/NORAD_CAT_ID asc/format/json`: every catalogue entry whose name contains QIANFAN.
2. The launches are the first eight characters of the international designator (`INTLDES[:8]`, for example `2024-140`) of those entries, plus any launch
   given with `--extra-launch` (used for launches whose payloads are still named `OBJECT x`).
3. For each launch: `class/satcat/INTLDES/~~<launch>/OBJECT_TYPE/PAYLOAD/format/json`.
4. Selection rule: a launch is kept whole if at least half of its payloads carry the name QIANFAN (or it was listed with `--extra-launch`); otherwise only
   the payloads named QIANFAN are kept.

Result on the 2026-09-24 run (checked): 248 payloads, all of type PAYLOAD and all named QIANFAN, in 15 launches: 2024-140 (18), 2024-185 (18), 2024-232 (18),
2025-016 (18), 2025-046 (18), 2025-233 (18), 2026-075 (18), 2026-104 (18), 2026-108 (18), 2026-121 (2), 2026-124 (18), 2026-125 (18), 2026-153 (18),
2026-155 (20), 2026-210 (10). Whether `--extra-launch` was used in the original download is not recorded. A reader querying later will find every payload of these launches named QIANFAN in the catalogue, so the name-based rule alone selects the same launches, but a query made while a launch still carries `OBJECT x` names needs `--extra-launch <launch>`.

## Step 2: element-set history (`class/gp_history`)

- Objects are requested in batches of 10 NORAD catalogue numbers, in ascending order:
  `class/gp_history/NORAD_CAT_ID/<id1>,...,<id10>/orderby/NORAD_CAT_ID asc,EPOCH asc/predicates/<predicates>/format/json`
- Predicates: `NORAD_CAT_ID, OBJECT_NAME, OBJECT_ID, OBJECT_TYPE, EPOCH, MEAN_MOTION, ECCENTRICITY, INCLINATION, RA_OF_ASC_NODE, ARG_OF_PERICENTER, MEAN_ANOMALY,
  BSTAR, MEAN_MOTION_DOT, MEAN_MOTION_DDOT, CREATION_DATE, GP_ID`.
- No epoch range is set in the query: the full history of each object is taken, so the start of the record is each object's first element set and the end is the
  date of the download (2026-09-24 for the record in the paper).

## Step 3: deduplication and derived fields (`common.load_gp`)

- Space-Track can hold several issues of the element set for one epoch. Records are sorted by `CREATION_DATE` and, for each (`NORAD_CAT_ID`, `EPOCH`), the newest
  issue is kept.
- Derived fields: semi-major axis from the mean motion with mu = 398,600.8 km^3/s^2 (WGS-72, the constants SGP4 uses), mean altitude above 6,378.135 km,
  perigee and apogee altitude, and `LAUNCH` = the first eight characters of `OBJECT_ID`.

Result (checked): 272,460 records downloaded, 260,106 element sets after deduplication, 248 objects, epochs from 2024-08-13 13:12 UTC to 2026-09-24 04:34 UTC.
These are the figures of Section 3.1 and Table 2.

## Optional queries (pilot stage only)

`--debris` adds the `gp_history` of the non-payload objects of launch 2024-140 and `--background` adds a snapshot of the current catalogue between 600 and
1,300 km (`class/gp/DECAY_DATE/null-val/EPOCH/>now-10/PERIAPSIS/<1300/APOAPSIS/>600`). They were used for pilot screens against other objects; the analyses in the
paper are between Qianfan satellites only, and the outputs of those pilot screens are not part of the release.

## Re-querying later

Space-Track can prune or supersede old element sets, and Qianfan launches continue, so a later query will contain more objects and epochs. The record of the paper
ends at 2026-09-24; the analysis scripts use the daily windows 2025-03-13 to 2026-09-22 (Section 3.4).
