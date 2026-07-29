"""Allocation-method overview: net Sharpe ratios for the United States.

Inputs: published complete-sample CSV files ``h1_per_market.csv`` (60/40,
    permanent portfolio and binary-switch Sharpe ratios) and
    ``h3_per_market.csv`` (the graded-variant Sharpe ratio). The two published
    values from the previous four-quadrant paper are constants stated in that
    paper.
Outputs: ``allocation_methods_overview_fr.png`` (French build) or the ``_en``
    variant, in the complete-sample result directory.
Purpose: place the H1 binary switch and the H3 graded variant on one net-Sharpe
    scale against the references and the previously published allocations.

The figure reads validated CSV only; it does not rerun the engine.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common.paths import FULL_SAMPLE_RESULTS
from figures.i18n import L, figpath
from figures.plot_style import BLUE, GREY, GRID, INK, ORANGE, PURPLE, setup_matplotlib

OUT = FULL_SAMPLE_RESULTS


def setup() -> None:
    setup_matplotlib()


def allocation_figure() -> None:
    h1 = pd.read_csv(OUT / "h1_per_market.csv").set_index("market")
    h3 = pd.read_csv(OUT / "h3_per_market.csv").set_index("market")
    usa_h1 = h1.loc["USA"]
    usa_h3 = h3.loc["USA"]

    bars = {
        L("60/40, référence", "60/40, reference"): (
            usa_h1["sharpe_6040"],
            GREY,
            "reference",
        ),
        L("Portefeuille permanent", "Permanent portfolio"): (
            usa_h1["sharpe_permanent"],
            GREY,
            "reference",
        ),
        L(
            "H1, bascule binaire obligations-or",
            "H1, binary bond-gold switch",
        ): (
            usa_h1["sharpe_switch"],
            BLUE,
            "h1",
        ),
        L("Variante graduée H3", "H3 graded variant"): (
            usa_h3["sharpe_amplitude"],
            ORANGE,
            "h3",
        ),
        L(
            "Browne et énergie statique, papier précédent",
            "Browne and static energy, previous paper",
        ): (
            0.57,
            PURPLE,
            "published",
        ),
        L(
            "Quatre quadrants dynamiques, papier précédent",
            "Dynamic four quadrants, previous paper",
        ): (
            0.64,
            PURPLE,
            "published",
        ),
        L(
            "Quatre quadrants inclinés, papier précédent",
            "Tilted four quadrants, previous paper",
        ): (
            0.70,
            PURPLE,
            "published",
        ),
    }
    order = sorted(bars, key=lambda label: bars[label][0])
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    for position, label in zip(y, order):
        value, color, kind = bars[label]
        hatch = "..." if kind == "published" else None
        ax.barh(
            position,
            value,
            height=0.64,
            color=color,
            alpha=0.92,
            hatch=hatch,
            edgecolor="white" if kind == "published" else "none",
            zorder=2,
        )
        ax.text(
            value + 0.006,
            position,
            f"{value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            color=INK,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 0.82)
    ax.set_xlabel(
        L(
            r"Ratio de Sharpe net, en excès du taux court et annualisé, "
            r"États-Unis",
            r"Net Sharpe ratio, in excess of the short rate and annualized, "
            r"United States",
        )
    )
    ax.legend(
        handles=[
            Patch(fc=BLUE, label=L("H1, bascule binaire", "H1, binary switch")),
            Patch(fc=ORANGE, label=L("Variante graduée H3", "H3 graded variant")),
            Patch(fc=GREY, label=L("Références", "References")),
            Patch(
                fc=PURPLE,
                hatch="...",
                ec="white",
                label=L(
                    "Papier précédent, valeurs publiées",
                    "Previous paper, published values",
                ),
            ),
        ],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=9.5,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(
        figpath(OUT / "allocation_methods_overview_fr.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    setup()
    allocation_figure()
    print("wrote allocation_methods_overview figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
