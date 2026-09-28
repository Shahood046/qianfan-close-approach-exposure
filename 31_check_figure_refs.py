"""Consistency check of figure references (run after any renumbering or figure edit).  No analysis is run.

Checks
  1. Every Fig./Figure reference in manuscript/03-12 resolves to an existing figure and, where a panel letter is given, to an existing panel.
  2. Every main figure (1-11) is referenced at least once in the manuscript body (sections 3-9).
  3. Fig. A1 is referenced only from the places meant to point to the appendix (Section 5.4 and the Section 5 notes).
  4. The dropped working figure has no reference: Fig. 7 is now the model-validation figure, so any 'Fig. 7' must sit in Section 5.
  5. manuscript/01 contains figure references only to cited papers (Handley 2018), which are never renumbered.
  6. FIGURE_CAPTIONS.md has one caption per existing figure file, in order, and every out/fig*.png named in the manuscript, captions, number files and
     figure scripts exists.
  7. No old (working-number) file or script name is left in the current documents and scripts.

Usage:  py 31_check_figure_refs.py       (exit status 1 if any check fails)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
MAIN = [str(i) for i in range(1, 12)]
FILES = {"1": "fig1_buildout", "2": "fig2_altitude_states", "3": "fig3_residuals", "4": "fig4_geometry", "5": "fig5_slot_grid",
         "6": "fig6_grid_offset", "7": "fig7_model_vs_sgp4", "8": "fig8_phase_trajectories", "9": "fig9_concentration",
         "10": "fig10_counterfactual", "11": "fig11_episodes_timeline", "A1": "figA1_appendix_projection"}
PANELS = {"4": "abc", "5": "ab", "8": "ab", "9": "ab", "A1": "abc"}
REF = re.compile(r"\b(Figs?\.|Figure)\s*(A1|\d+)([a-c](?:,\s*[a-c])?)?")
OLD_NAMES = ["fig8_model_vs_sgp4", "fig10_phase_trajectories", "fig11_concentration", "fig12_counterfactual", "fig13_episodes_timeline",
             "fig9_appendix_projection", "fig10_picks", "25_fig8_model", "26_fig11_panels", "27_fig10_trajectories", "29_appendix_fig9",
             "FIG8_MODEL", "FIG10_TRAJECTORIES", "FIG11_PANELS", "fig_phase_trajectories", "residuals_vs_age"]
# documents that keep older numbers on purpose (history, legacy pilot material, superseded outputs)
HISTORICAL = {"SESSION_LOG_2026-09-28.md", "HANDOFF.md", "FINDINGS.md", "PAPER_OUTLINE.md", "LITERATURE.md", "FIGURE_AUDIT.md", "README.md",
              "06_figures.py", "07_crossing_model.py", "10_episodes.py", "03_residuals.py", "31_check_figure_refs.py",
              "RELEASE_MANIFEST.md"}
fails: list[str] = []


def fail(msg):
    fails.append(msg)
    print("FAIL:", msg)


def text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def refs(p: Path):
    for n, line in enumerate(text(p).splitlines(), 1):
        for m in REF.finditer(line):
            yield n, line, m.group(2), (m.group(3) or "").replace(",", "").replace(" ", "")


HAVE_MS = (ROOT / "manuscript").exists() and (ROOT / "FIGURE_CAPTIONS.md").exists()   # the public release has neither
if not HAVE_MS:
    print("manuscript/ or FIGURE_CAPTIONS.md not present (public release): checking only the figure files and the names used in scripts and documents")
ms = sorted((ROOT / "manuscript").glob("*.md")) if HAVE_MS else []
seen = {k: [] for k in FILES}
for p in ms:
    if p.name.startswith("01_"):
        for n, line, num, pan in refs(p):
            if "Handley" not in line:
                fail(f"{p.name}:{n}: figure reference not tied to a cited paper: {line[:90]}")
        continue
    for n, line, num, pan in refs(p):
        if num not in FILES:
            fail(f"{p.name}:{n}: Fig. {num} does not exist")
            continue
        for ch in pan:
            if ch not in PANELS.get(num, ""):
                fail(f"{p.name}:{n}: Fig. {num}{ch}: no such panel")
        seen[num].append((p.name, n))

body = {k: [x for x in v if re.match(r"0[3-9]_", x[0])] for k, v in seen.items()}
for k in MAIN if HAVE_MS else []:
    if not body[k]:
        fail(f"main Fig. {k} is not referenced in sections 3-9")
for name, n in seen["A1"]:
    if not name.startswith("05_"):
        fail(f"Fig. A1 referenced outside Section 5: {name}:{n}")
for name, n in seen["7"]:
    if not name.startswith("05_"):
        fail(f"Fig. 7 referenced outside Section 5 (working Fig. 7 was dropped): {name}:{n}")
for k, f in FILES.items():
    if not (OUT / f"{f}.png").exists():
        fail(f"missing figure file out/{f}.png")

cap = text(ROOT / "FIGURE_CAPTIONS.md") if HAVE_MS else ""
if HAVE_MS:
    labels = re.findall(r"^\*\*Fig\. (A1|\d+)\*\*", cap, re.M)
    if labels != MAIN + ["A1"]:
        fail(f"FIGURE_CAPTIONS.md labels are {labels}")
    for num, f in re.findall(r"^\*\*Fig\. (A1|\d+)\*\* \(.*?`out/(fig\w+)\.png`", cap, re.M):
        if FILES[num] != f:
            fail(f"caption of Fig. {num} names {f}, expected {FILES[num]}")

named = set()
for p in list(ROOT.glob("*.py")) + list(ROOT.glob("*.md")) + ms:
    if p.name in HISTORICAL:
        continue
    for m in re.finditer(r"out/(fig[0-9A]\w*?)\.png", text(p)):
        named.add((p.name, m.group(1)))
    t = text(p)
    for old in OLD_NAMES:
        if old in t:
            fail(f"{p.name}: old name '{old}' still present")
for pname, f in sorted(named):
    if not (OUT / f"{f}.png").exists():
        fail(f"{pname}: names out/{f}.png, which does not exist")

print(f"figures referenced in the manuscript body: " + ", ".join(f"{k}:{len(v)}" for k, v in body.items()))
print("A1 referenced from:", seen["A1"], "| Fig. 7 referenced from:", seen["7"])
print("RESULT:", "all checks passed" if not fails else f"{len(fails)} problem(s)")
sys.exit(1 if fails else 0)
