# Qianfan phasing and close-approach exposure: analysis code and derived tables

Code, derived tables and figures for the paper *Phase errors and close-approach exposure in the Qianfan constellation: an empirical
reconstruction from public orbital data* (in preparation). The paper reconstructs the as-operated phasing of the Qianfan constellation from public
Space-Track element sets and relates it to the distribution of close approaches between Qianfan satellites. The analysis covers close approaches
between Qianfan satellites only, uses public element sets (no covariance, so no collision probability), and describes reconstructed structure and
statistical association, not operator design or intent.

## What is and is not in this repository

| Included | Not included |
|---|---|
| Processing code, `01_fetch.py` to `32_build_release.py`, and `common.py` | Space-Track element sets in any form (raw, cached or reformatted) |
| The query specification (`QUERY_SPEC.md`) | Credentials (`.env`) |
| Derived result tables (`*_CLAIMS.md`, `MARGIN_TABLE.md`, `COUNTERFACTUAL.md`, `EPISODES_TIME.md`, ... and CSV tables in `out/`) | Per-satellite, per-epoch orbital-element tables; the pair-window pickles (`data/`) |
| Rendered figures (`out/fig*.png`) and the scripts that draw them | Screens against other catalogued objects (pilot stage), per-pass encounter dumps, residual pairs |
| `out/superseded/`: pilot-era figures and outputs, kept for the record and **not valid for the manuscript** | Working notes and manuscript drafts |

Space-Track's User Agreement does not allow redistribution of its data. Anyone reproducing the results downloads the element sets with their own
Space-Track account (free) and accepts its terms; see `QUERY_SPEC.md`.

## Setup

Python 3.14 and the packages in `requirements.txt` (exact versions used for the results):

```bash
py -m pip install -r requirements.txt
```

Credentials are read from the environment (or from a `.env` file next to the scripts, which is git-ignored and never released):

```powershell
$env:SPACETRACK_USER = "you@example.com"
$env:SPACETRACK_PASS = "..."
```

## Order of the pipeline

The numeric prefix gives the order; each script's docstring states its prerequisites, inputs, outputs and the manuscript section it supports.

| Scripts | Purpose | Sections |
|---|---|---|
| `01_fetch.py` | Space-Track download into `data/` (not released) | 3.1 |
| `02_states.py` | orbital-transition states, arrival epochs, band-width variants (`--X 10 --tag _X10`, `--X 30 --tag _X30`) | 3.2 |
| `03_residuals.py` | SGP4 prediction residuals by state and propagation age | 3.3 |
| `04_screen.py`, `04b_merge_daily.py` | close-approach screens (3-day and daily cadence) and the merge of the two daily screens | 3.4 |
| `05_planes.py` to `09_model_checks.py` | planes, crossing model, robustness of the slot grid, model against SGP4 | 4, 5 |
| `10_episodes.py` to `16_bandwidth_check.py` | episodes, per-angle and per-margin exposure tables, sensitivity, cadence check, `LOCKED_CLAIMS.md` (`14`) | 7 to 9 |
| `17_geometry.py` to `21_signed_deviation.py` | reconstructed geometry, the 2026 rise, phase dynamics, phase-variable comparison, signed pair deviation | 4, 6, 7 |
| `22_margin_table_fig.py` to `24_episodes_time.py` | Table 10, counterfactual, episodes over time | 7 to 9 |
| `25`, `26`, `27`, `28`, `29`, `30` (figure scripts) | figures (table below); they draw from the tables the scripts above save | all |
| `31_check_figure_refs.py` | checks that every figure reference in the manuscript resolves | |
| `32_build_release.py` | assembles the release tree in `release/` and checks it | |

## Figures

| Figure | File | Script |
|---|---|---|
| 1 build-out by state | `out/fig1_buildout.png` | `30_data_figs.py` |
| 2 altitude histories by state | `out/fig2_altitude_states.png` | `30_data_figs.py` |
| 3 SGP4 residuals by state | `out/fig3_residuals.png` | `30_data_figs.py` |
| 4 reconstructed geometry | `out/fig4_geometry.png` | `28_geometry_figs.py` |
| 5 in-plane slot grid | `out/fig5_slot_grid.png` | `28_geometry_figs.py` |
| 6 grid offset at the crossing | `out/fig6_grid_offset.png` | `28_geometry_figs.py` |
| 7 crossing model against SGP4 | `out/fig7_model_vs_sgp4.png` | `25_fig7_model.py` |
| 8 phase-error trajectories | `out/fig8_phase_trajectories.png` | `27_fig8_trajectories.py` |
| 9 margin population and event concentration | `out/fig9_concentration.png` | `26_fig9_panels.py` |
| 10 counterfactual | `out/fig10_counterfactual.png` | `23_counterfactual.py` |
| 11 recurring episodes | `out/fig11_episodes_timeline.png` | `24_episodes_time.py` |
| A1 phase variable and crossing projection (appendix) | `out/figA1_appendix_projection.png` | `29_appendix_figA1.py` |

The figure scripts read the saved result tables (`out/*.csv`) or, for Figs. 9 to 11, the pair-window table `data/pair_windows_daily.pkl.gz` built by
`11_per_angle.py`; the numbers behind each figure are written next to it (`FIG*_NUMBERS.md`, `FIG7_MODEL.md`, `FIG8_TRAJECTORIES.md`, `FIG9_PANELS.md`).

## Reproducibility and its limits

- Exact numerical reproduction needs a fresh Space-Track query. The record of the paper ends on 2026-09-24; a later query contains more objects and epochs,
  and Space-Track may prune or supersede old element sets, so results may differ in the last digits and, for later windows, in content.
- Checks made when this release was assembled (2026-09-28): every script compiles; the figure scripts were re-run from the authors' local copies of the
  saved tables and of the pair-window table (Figs. 1 to 8 and A1 read the saved tables; Figs. 9 to 11 also recompute their statistics) and reproduce the locked numbers; the query specification was checked against the stored data (272,460 records, 260,106 element sets after
  deduplication, 248 objects); the merged daily screen was compared with the two screens it is built from (`04b_merge_daily.py --check`).
- Not done: the whole pipeline was **not** re-run from raw element sets during release preparation, because the raw data cannot be included here. The
  regenerated result files were compared with the previous versions where a script was re-run; differences were limited to wording and figure names.
- The merged daily screen `out/encounter_pairs_daily.csv` was first produced by a manual merge. `04b_merge_daily.py` reproduces it as a set of rows (after sorting on the
  pair key, every column is identical except four floating-point columns that differ by round-off, at most 3.6e-15). The original row order was not preserved. This does not
  matter downstream: the analyses impose their own ordering (sorted window list, group-by reductions, merges on keys) and no step uses positional access, shift, diff,
  rolling, head or tail on the screen (audit in the docstring of `04b_merge_daily.py`).

## Terminology of the results

Reconstructed nominal geometry, reconstructed as-operated phasing and observed phase deviation describe what the public element sets show. None of the tables
or figures identifies an operator's design, control law or intent.

## Licence and citation

Code: MIT (`LICENSE`). Derived tables, figures and documentation: CC BY 4.0 (`LICENSE-DATA.md`). Raw Space-Track orbital data is not redistributed and remains under
the Space-Track User Agreement, independently of these licences. Citation: `CITATION.cff`.

## Authors

Shahood Sajid, Ibrar Junaid, Abdul Moeed.

Repository: https://github.com/Shahood046/qianfan-close-approach-exposure
