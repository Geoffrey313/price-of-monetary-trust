"""Regenerate the two market-by-market monetary-state figures.

Inputs: published monthly-state and per-market H1 CSV files.
Outputs: the signal heatmap and implemented-state timeline PNG files.
Purpose: accompany the temporal H1 analysis with figures ordered by published
H1 advantage.

This script redraws only the signal heatmap and the implemented-state timeline,
reading the committed monthly-state CSV rather than re-running the engine, so the
figures reproduce from data already in ``results/`` without the raw pipeline.

The single change relative to the engine-based drawings in
``plot_complete_sample_paper.signal_heatmap`` and
``analyze_signal_state_pre_post_2000.figure_c`` is the row order: markets are
sorted by their H1 advantage (Sharpe difference against the binding benchmark),
largest at the top and smallest at the bottom, so every row axis reads
monotonically. The styling, colours and layout mirror those functions.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from common.paths import FULL_SAMPLE_RESULTS
from figures.i18n import L, figpath, market_name
from figures.plot_style import (
    BLUE,
    GOLD,
    GRID,
    INK,
    ORANGE,
    OUTSIDE,
    SURFACE,
    setup_matplotlib,
)

OUT = FULL_SAMPLE_RESULTS
MONTHLY = OUT / "h1_signal_states_monthly_2026-07-27.csv"
PER_MARKET = OUT / "h1_per_market.csv"
CUT = pd.Period("2000-01", freq="M")

def setup() -> None:
    setup_matplotlib()


def advantage_order() -> tuple[list[str], dict[str, str]]:
    """Markets sorted by H1 advantage, largest first (top row)."""
    per = pd.read_csv(PER_MARKET)
    per = per.sort_values("advantage", ascending=False).reset_index(drop=True)
    order = per["market"].tolist()
    names = {
        code: market_name(code, country)
        for code, country in zip(per["market"], per["country"])
    }
    return order, names


def state_matrix(order: list[str]) -> tuple[pd.DataFrame, pd.PeriodIndex]:
    data = pd.read_csv(MONTHLY)[["market", "month", "bond_state"]].copy()
    data["period"] = pd.PeriodIndex(data["month"], freq="M")
    periods = pd.period_range(data["period"].min(), data["period"].max(), freq="M")
    matrix = (
        data.pivot(index="market", columns="period", values="bond_state")
        .reindex(index=order, columns=periods)
    )
    return matrix, periods


def heatmap(order: list[str], names: dict[str, str]) -> None:
    matrix, periods = state_matrix(order)
    values = matrix.to_numpy(dtype=float)
    dates = periods.to_timestamp()
    fig, (ax, lower) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.1),
        sharex=True,
        gridspec_kw={"height_ratios": [4.8, 1], "hspace": 0.08},
    )
    cmap = ListedColormap([GOLD, BLUE])
    cmap.set_bad(OUTSIDE)
    ax.imshow(
        np.ma.masked_invalid(values),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[
            mdates.date2num(dates[0]),
            mdates.date2num(dates[-1]),
            len(order) - 0.5,
            -0.5,
        ],
    )
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[m] for m in order])
    ax.legend(
        handles=[
            Patch(fc=BLUE, label=L("obligations", "bonds")),
            Patch(fc=GOLD, label=L("or", "gold")),
            Patch(fc=OUTSIDE, label=L("hors fenêtre", "out of window")),
        ],
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0, 1.11),
    )
    valid = np.sum(~np.isnan(values), axis=0)
    gold_fraction = np.divide(
        np.nansum(values == 0, axis=0),
        valid,
        out=np.full(len(valid), np.nan),
        where=valid > 0,
    )
    lower.fill_between(dates, 0, gold_fraction, color=GOLD, alpha=0.25)
    lower.plot(dates, gold_fraction, color=GOLD, lw=1.2)
    lower.set_ylim(0, 1)
    lower.set_yticks([0, 0.5, 1])
    lower.set_yticklabels(["0", "50", "100"])
    lower.set_ylabel(L(r"marchés en or (\%)", r"markets in gold (\%)"), fontsize=9)
    lower.xaxis.set_major_locator(mdates.YearLocator(10))
    lower.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lower.grid(axis="y", color=GRID, lw=0.7)
    for axis in (ax, lower):
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(figpath(OUT / "currency_signal_heatmap_13_markets.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def timeline(order: list[str], names: dict[str, str]) -> None:
    matrix, periods = state_matrix(order)
    month_starts = periods.to_timestamp(how="start")
    right_edge = (periods[-1] + 1).to_timestamp(how="start")
    x_edges = mdates.date2num(month_starts.to_pydatetime()).tolist()
    x_edges.append(mdates.date2num(right_edge.to_pydatetime()))
    y_edges = np.arange(len(order) + 1)

    fig, ax = plt.subplots(figsize=(11.8, 6.8))
    cmap = ListedColormap([ORANGE, BLUE])
    cmap.set_bad(SURFACE)
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.pcolormesh(
        x_edges,
        y_edges,
        np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
        cmap=cmap,
        norm=norm,
        shading="flat",
        rasterized=True,
    )
    cut_date = CUT.to_timestamp(how="start")
    ax.axvline(cut_date, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=3)
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([names[m] for m in order])
    ax.invert_yaxis()
    ax.set_xlim(month_starts[0], right_edge)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel(L(r"mois", r"month"))
    ax.set_axisbelow(False)
    for boundary in range(1, len(order)):
        ax.axhline(boundary, color=SURFACE, lw=1.0, zorder=2)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor="none", label=L(r"état obligations", r"bond state")),
            Patch(facecolor=ORANGE, edgecolor="none", label=L(r"état or", r"gold state")),
            Line2D([], [], color=INK, lw=1.1, ls=(0, (4, 3)), label=L(r"janvier 2000", r"January 2000")),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.subplots_adjust(left=0.15, right=0.99, bottom=0.20, top=0.99)
    fig.savefig(figpath(OUT / "h1_signal_state_timeseries_13_markets_fr.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    setup()
    order, names = advantage_order()
    heatmap(order, names)
    timeline(order, names)
    print("regenerated (top -> bottom, largest -> smallest H1 advantage):")
    print("  " + " ".join(order))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
