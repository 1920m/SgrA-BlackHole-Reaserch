"""Generate the explanatory matplotlib schematics for the thesis
(paper/THESIS_PLAN.md §2.2b) into paper/figures/.

Same toolchain and conventions as make_figures.py: matplotlib only, no TikZ,
no hand-drawn or AI-generated content; deterministic (fixed seeds) so every
diagram is reproducible by `make figures`.

Run: py -3.11 scripts/make_diagrams.py
"""
from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "font.family": "serif",
    "axes.linewidth": 0.7,
    "figure.dpi": 150,
})

# Diagram functions land here as the thesis chapters are written
# (dg_shadow_rays, dg_kerr_anatomy, dg_kepler_ellipse, dg_rosette,
#  dg_three_redshifts, dg_frame_drag, dg_chi2_sketch, dg_mcmc_walkers,
#  dg_quadrupole, dg_scales, dg_timeline, ...).
DIAGRAMS: list = []


def main() -> None:
    print("generating thesis diagrams ...")
    for fn in DIAGRAMS:
        fn()
    print(f"done ({len(DIAGRAMS)} diagrams).")


if __name__ == "__main__":
    main()
