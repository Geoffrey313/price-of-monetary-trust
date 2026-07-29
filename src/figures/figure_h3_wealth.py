"""Relative-wealth grid: the H3 graded variant against the binary switch.

Inputs: the wired H1 engine series (binary switch) from
    ``engine.engine.run_market`` and the graded-variant return series from
    ``analysis.h3_variant`` (``series_for`` and ``run_variant``), built from the
    configured snapshots or the derived layer under
    ``FOUR_QUADRANT_FROM_DERIVED=1``.
Outputs: ``h3_wealth_relative_13_markets_fr.png`` (French build) or the ``_en``
    variant, in the complete-sample result directory.
Purpose: show, market by market, cumulative wealth of the graded variant divided
    by cumulative wealth of the binary switch. The ratio drifts below one in
    every market, so the retained binary form is not improved by grading.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from analysis import run_full_sample as R
from analysis.h3_variant import run_variant, series_for
from common.names import ENGLISH_NAMES, NAMES
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E
from figures.i18n import L, figpath
from figures.plot_style import GRID, INK, ORANGE, setup_matplotlib

OUT = FULL_SAMPLE_RESULTS


def setup() -> None:
    setup_matplotlib()


def wealth(returns: pd.Series) -> pd.Series:
    return 100 * (1 + returns).cumprod()


def collect() -> dict[str, dict[str, pd.Series]]:
    R.wire_everything()
    E.run_market.return_series = True
    out: dict[str, dict[str, pd.Series]] = {}
    for market in R.MARKETS:
        try:
            series = E.run_market(market, real_gate=False, bootstrap_draws=0)["series"]
            returns, _sig_bin, frac = series_for(market)
            amplitude = run_variant(returns, frac)
        except Exception as error:  # noqa: BLE001 - report and skip a bad market
            print(f"  [skip] {market}: {str(error)[:70]}")
            continue
        index = series["strat"].index.intersection(amplitude.index)
        out[market] = {
            "binary": series["strat"].reindex(index),
            "amp": amplitude.reindex(index),
        }
    return out


def relative_grid(results: dict[str, dict[str, pd.Series]]) -> None:
    order = sorted(
        results,
        key=lambda m: (
            wealth(results[m]["amp"]) / wealth(results[m]["binary"])
        ).iloc[-1],
    )
    fig, axes = plt.subplots(4, 4, figsize=(16, 13))
    axes = axes.ravel()
    for ax, market in zip(axes, order):
        data = results[market]
        relative = wealth(data["amp"]) / wealth(data["binary"])
        dates = relative.index.to_timestamp()
        ax.axhline(1.0, color=INK, lw=0.9, zorder=1)
        ax.plot(dates, relative.values, color=ORANGE, lw=1.8, zorder=2)
        ax.fill_between(
            dates,
            relative.values,
            1.0,
            where=(relative.values <= 1.0),
            color=ORANGE,
            alpha=0.15,
        )
        ax.set_title(
            L(NAMES[market], ENGLISH_NAMES[market]),
            fontsize=12,
            loc="left",
            pad=5,
        )
        ax.tick_params(labelsize=9, length=0)
        ax.grid(True, color=GRID, lw=0.7)
        ax.xaxis.set_major_locator(mdates.YearLocator(10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    for position in range(len(order), len(axes)):
        axes[position].axis("off")
    legend_axis = axes[len(order)] if len(order) < len(axes) else None
    if legend_axis is not None:
        legend_axis.plot(
            [],
            [],
            color=ORANGE,
            lw=2.4,
            label=L(
                r"Richesse graduée / richesse binaire",
                r"Graded wealth / binary wealth",
            ),
        )
        legend_axis.legend(loc="center", frameon=False, fontsize=11)
    fig.supxlabel(L(r"Année", r"Year"), fontsize=12)
    fig.supylabel(
        L(
            r"Rapport de richesse cumulée, graduée / binaire",
            r"Cumulative wealth ratio, graded / binary",
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=(0.03, 0.03, 1, 1))
    fig.savefig(
        figpath(OUT / "h3_wealth_relative_13_markets_fr.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    setup()
    results = collect()
    relative_grid(results)
    print("wrote h3_wealth_relative_13_markets figure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
