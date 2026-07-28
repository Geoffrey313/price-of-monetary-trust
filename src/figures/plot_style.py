"""Apply the shared publication style to every Matplotlib figure.

Inputs: none.
Outputs: updated Matplotlib runtime parameters; figure modules write PNG files
to the complete-sample result directory.
Purpose: keep the paper's H1, state, and era figures visually consistent.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

BLUE = "#2166ac"
ORANGE = "#b35806"
GREY = "#737373"
LIGHT_GREY = "#b7b7b7"
GOLD = "#b58b2a"
PURPLE = "#762a83"
RED = "#b2182b"
GRID = "#e2e2e2"
INK = "#111111"
SURFACE = "#ffffff"
OUTSIDE = "#f2f2ef"


def setup_matplotlib() -> None:
    """Install the paper's LaTeX-compatible serif plotting style."""
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "xtick.color": "#444444",
            "ytick.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
        }
    )
