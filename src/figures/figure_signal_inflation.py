"""Cross-market inflation gap between the bond state and the gold state.

Inputs: published complete-sample CSV/JSON files
    ``inflation_signal_by_market.csv`` (per-market bond-state inflation
    coefficient and HAC standard error), ``summary.json`` (the pooled
    ``inflation_validity`` block) and
    ``inflation_signal_joint_inference_2026-07-27.csv`` (Driscoll--Kraay
    pooled confidence interval).
Outputs: ``currency_signal_inflation_13_markets_fr.png`` (French build) or the
    ``_en`` variant, in the complete-sample result directory.
Purpose: show, per market and pooled, how much higher inflation runs in the
    bond state than in the gold state, with the pooled Driscoll--Kraay interval.

The figure reads validated CSV/JSON only; it does not rerun the engine or any
inference procedure.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from common.names import ENGLISH_NAMES, NAMES
from common.paths import FULL_SAMPLE_RESULTS
from figures.i18n import L, figpath
from figures.plot_style import BLUE, GOLD, GRID, INK, setup_matplotlib

OUT = FULL_SAMPLE_RESULTS


def setup() -> None:
    setup_matplotlib()


def inflation_figure() -> None:
    national = pd.read_csv(OUT / "inflation_signal_by_market.csv")
    with (OUT / "summary.json").open(encoding="utf-8") as stream:
        pooled = json.load(stream)["inflation_validity"]
    joint = pd.read_csv(
        OUT / "inflation_signal_joint_inference_2026-07-27.csv"
    ).set_index("method")
    pooled_dk12 = joint.loc["driscoll_kraay_12"]

    rows = national.assign(
        country=national["market"].map(lambda m: L(NAMES[m], ENGLISH_NAMES[m])),
        low=national["coefficient_bond_state"] - 1.96 * national["se_hac12"],
        high=national["coefficient_bond_state"] + 1.96 * national["se_hac12"],
        pooled=False,
    )[
        ["market", "country", "coefficient_bond_state", "low", "high", "pooled"]
    ]
    pooled_row = pd.DataFrame(
        [
            {
                "market": "POOLED",
                "country": L(
                    "Groupé, effets fixes de marché",
                    "Pooled, market fixed effects",
                ),
                "coefficient_bond_state": pooled["pooled_beta"],
                "low": pooled_dk12["ci95_low"],
                "high": pooled_dk12["ci95_high"],
                "pooled": True,
            }
        ]
    )
    data = pd.concat([rows, pooled_row], ignore_index=True).sort_values(
        "coefficient_bond_state", ascending=False
    )

    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    for position, row in zip(y, data.itertuples()):
        color = BLUE if row.pooled else GOLD
        ax.plot([row.low, row.high], [position, position], color=color, lw=1.4)
        ax.scatter(
            row.coefficient_bond_state,
            position,
            s=65 if row.pooled else 45,
            color=color,
            zorder=3,
        )
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(data["country"])
    ax.set_xlabel(
        L(
            r"Inflation dans l'état obligataire moins l'état or "
            r"(points de pourcentage)",
            r"Inflation in the bond state minus the gold state "
            r"(percentage points)",
        )
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.legend(
        handles=[
            Line2D([], [], color=GOLD, marker="o", label=L("Marché", "Market")),
            Line2D(
                [],
                [],
                color=BLUE,
                marker="o",
                label=L(
                    r"Estimation groupée, IC Driscoll--Kraay",
                    r"Pooled estimate, Driscoll--Kraay CI",
                ),
            ),
        ],
        loc="upper right",
        frameon=False,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(
        figpath(OUT / "currency_signal_inflation_13_markets_fr.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    setup()
    inflation_figure()
    print("wrote currency_signal_inflation_13_markets figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
