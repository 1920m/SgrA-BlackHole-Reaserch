"""Shared matplotlib style for all report figures and diagrams
(paper/THESIS_PLAN.md §2.2a): one palette, one font scale, applied by both
make_figures.py (data plots) and make_diagrams.py (schematics).

Palette: Okabe & Ito's colourblind-safe set.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colourblind-safe palette
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
}
# Ordered cycle for multi-series plots
CYCLE = [OKABE[k] for k in
         ("blue", "vermillion", "green", "orange", "purple", "sky", "black")]


def apply_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "legend.fontsize": 8,
        "legend.framealpha": 0.9,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi": 150,
    })
