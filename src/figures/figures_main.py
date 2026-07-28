"""Generate every sample-sensitive main paper figure.

Inputs: published complete-sample CSV files and the frozen engine inputs.
Outputs: the main PNG figures in the complete-sample result directory.
Purpose: render the cross-market H1--H3, risk, wealth, era, and sensitivity
figures referenced by the paper.

The files are kept in the dated rerun directory. The manuscript can therefore
reference the audited figures without overwriting historical graphics.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from analysis import run_full_sample as R
from engine import engine as E
from figures.plot_style import (
    BLUE,
    GOLD,
    GREY,
    GRID,
    INK,
    LIGHT_GREY,
    ORANGE,
    PURPLE,
    SURFACE,
    setup_matplotlib,
)

OUT = R.OUT
CUT = pd.Period("1999-01", "M")


def setup() -> None:
    setup_matplotlib()


def collect() -> dict[str, dict]:
    R.wire_everything()
    E.run_market.return_series = True
    return {
        market: E.run_market(market, real_gate=False, bootstrap_draws=0)
        for market in R.MARKETS
    }


def best(series: dict[str, pd.Series]) -> tuple[str, pd.Series]:
    return R.harder_benchmark(series)


def signal_heatmap(results: dict[str, dict]) -> None:
    periods = pd.period_range(
        min(results[m]["series"]["strat"].index.min() for m in R.MARKETS),
        max(results[m]["series"]["strat"].index.max() for m in R.MARKETS),
        freq="M",
    )
    order = (
        pd.read_csv(OUT / "h1_per_market.csv")
        .sort_values("advantage", ascending=False)["market"]
        .tolist()
    )
    matrix = np.full((len(order), len(periods)), np.nan)
    for row, market in enumerate(order):
        signal = results[market]["series"]["signal"].reindex(periods)
        valid = results[market]["series"]["strat"].index
        signal = signal.where(signal.index.isin(valid))
        matrix[row] = signal.values
    dates = periods.to_timestamp()
    fig, (ax, lower) = plt.subplots(
        2,
        1,
        figsize=(11.5, 7.1),
        sharex=True,
        gridspec_kw={"height_ratios": [4.8, 1], "hspace": 0.08},
    )
    cmap = ListedColormap([GOLD, BLUE])
    cmap.set_bad("#f2f2ef")
    ax.imshow(
        np.ma.masked_invalid(matrix),
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
    ax.set_yticklabels([R.NAMES[m] for m in order])
    ax.legend(
        handles=[
            Patch(fc=BLUE, label="obligations"),
            Patch(fc=GOLD, label="or"),
            Patch(fc="#f2f2ef", label="hors fenêtre"),
        ],
        loc="upper left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0, 1.11),
    )
    valid = np.sum(~np.isnan(matrix), axis=0)
    gold_fraction = np.divide(
        np.nansum(matrix == 0, axis=0),
        valid,
        out=np.full(len(valid), np.nan),
        where=valid > 0,
    )
    lower.fill_between(dates, 0, gold_fraction, color=GOLD, alpha=0.25)
    lower.plot(dates, gold_fraction, color=GOLD, lw=1.2)
    lower.set_ylim(0, 1)
    lower.set_yticks([0, 0.5, 1])
    lower.set_yticklabels(["0", "50", "100"])
    lower.set_ylabel(r"marchés en or (\%)", fontsize=9)
    lower.xaxis.set_major_locator(mdates.YearLocator(10))
    lower.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    lower.grid(axis="y", color=GRID, lw=0.7)
    for axis in (ax, lower):
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "currency_signal_heatmap_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def hypothesis_ladder() -> None:
    h1 = pd.read_csv(OUT / "h1_per_market.csv")
    h3 = pd.read_csv(OUT / "h3_per_market.csv")[
        ["market", "sharpe_amplitude"]
    ]
    data = h1.merge(h3, on="market")
    data["sharpe_benchmark"] = np.maximum(
        data["sharpe_6040"], data["sharpe_permanent"]
    )
    data = data.sort_values("advantage").reset_index(drop=True)
    y = np.arange(len(data))[::-1]
    fig, ax = plt.subplots(figsize=(9.6, 7.0))
    for position, row in zip(y, data.itertuples()):
        values = [row.sharpe_benchmark, row.sharpe_amplitude, row.sharpe_switch]
        ax.plot(
            [min(values), max(values)],
            [position, position],
            color=GRID,
            lw=2.4,
            zorder=1,
        )
        ax.scatter(
            row.sharpe_benchmark,
            position,
            s=58,
            color=GREY,
            label="meilleure référence" if position == y[0] else None,
            zorder=3,
        )
        ax.scatter(
            row.sharpe_switch,
            position,
            s=60,
            color=BLUE,
            label="bascule binaire" if position == y[0] else None,
            zorder=3,
        )
        ax.scatter(
            row.sharpe_amplitude,
            position,
            s=60,
            color=ORANGE,
            label="variante graduée" if position == y[0] else None,
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([R.NAMES[m] for m in data["market"]])
    ax.set_xlabel(r"ratio de Sharpe net")
    ax.legend(frameon=False, loc="lower right")
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "hypotheses_ladder_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def jackknife() -> None:
    data = pd.read_csv(OUT / "h1_jackknife.csv").sort_values("alpha_annualized")
    full = float(
        pd.read_json(OUT / "summary.json", typ="series")["h1"]["risk_adjusted"][
            "alpha_annualized_sharpe"
        ]
    )
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    ax.axvspan(
        data["alpha_annualized"].min(),
        data["alpha_annualized"].max(),
        color=BLUE,
        alpha=0.08,
    )
    ax.axvline(full, color=GREY, ls="--", lw=1.1, label=rf"échantillon complet ${full:+.3f}$")
    ax.scatter(data["alpha_annualized"], y, color=BLUE, s=55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([rf"sans {R.NAMES[m]}" for m in data["omitted_market"]])
    ax.set_xlabel(r"$\widehat{\alpha}$ ajusté du risque, annualisé")
    ax.legend(frameon=False)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "h1_jackknife_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def risk_figure() -> None:
    data = pd.read_csv(OUT / "h1_per_market.csv").sort_values("vol_6040")
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 7.0), sharey=True)
    for ax, switch_col, bench_col, xlabel in (
        (
            axes[0],
            "vol_switch",
            "vol_6040",
            r"volatilité annualisée des rendements excédentaires (\%)",
        ),
        (
            axes[1],
            "drawdown_switch",
            "drawdown_6040",
            r"perte maximale (\%)",
        ),
    ):
        switch = data[switch_col].to_numpy(float) * 100
        bench = data[bench_col].to_numpy(float) * 100
        for position, left, right in zip(y, switch, bench):
            ax.plot([left, right], [position, position], color=GRID, lw=2.2)
        ax.scatter(
            bench,
            y,
            color=GREY,
            s=52,
            label="$60/40$ local",
            zorder=3,
        )
        ax.scatter(
            switch,
            y,
            color=BLUE,
            s=52,
            label="bascule binaire",
            zorder=3,
        )
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, color=GRID, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([R.NAMES[m] for m in data["market"]])
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "h1_risk_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def static_timing() -> None:
    data = pd.read_csv(OUT / "h1_static_timing.csv")
    labels = {
        "b6040": "$60/40$",
        "pp": "permanent",
        "matched": "statique comparable",
        "strat": "bascule dynamique",
    }
    colours = {
        "b6040": GREY,
        "pp": LIGHT_GREY,
        "matched": GOLD,
        "strat": BLUE,
    }
    metrics = [
        ("sharpe", r"ratio de Sharpe net"),
        ("volatility", r"volatilité annualisée"),
        ("max_drawdown", r"perte maximale"),
    ]
    methods = ["b6040", "pp", "matched", "strat"]
    fig, axes = plt.subplots(1, 3, figsize=(12.7, 5.7))
    rng = np.random.default_rng(R.SEED)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for x, method in enumerate(methods):
            values = data.loc[data["method"] == method, metric].to_numpy(float)
            jitter = rng.uniform(-0.08, 0.08, len(values))
            ax.scatter(
                np.full(len(values), x) + jitter,
                values,
                color=colours[method],
                alpha=0.4,
                s=24,
                edgecolor="none",
            )
            median = float(np.median(values))
            ax.plot(
                [x - 0.22, x + 0.22],
                [median, median],
                color=colours[method],
                lw=4,
                solid_capstyle="round",
            )
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels([labels[m] for m in methods], rotation=24, ha="right")
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, color=GRID, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT / "h1_static_timing_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def wealth_figures(results: dict[str, dict]) -> None:
    order = (
        pd.read_csv(OUT / "h1_per_market.csv")
        .sort_values("advantage", ascending=False)["market"]
        .tolist()
    )
    fig, axes = plt.subplots(4, 4, figsize=(13.2, 11.3))
    handles = None
    for ax, market in zip(axes.ravel(), order):
        series = results[market]["series"]
        benchmark_name, benchmark = best(series)
        switch_w = (1 + series["strat"]).cumprod() * 100
        bench_w = (1 + benchmark).cumprod() * 100
        ax.plot(
            switch_w.index.to_timestamp(),
            switch_w,
            color=BLUE,
            lw=1.4,
            label="bascule",
        )
        ax.plot(
            bench_w.index.to_timestamp(),
            bench_w,
            color=GREY,
            lw=1.1,
            label=benchmark_name,
        )
        ax.set_yscale("log")
        ax.text(
            0.03,
            0.95,
            R.NAMES[market],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7)
        handles = ax.get_legend_handles_labels()
    for ax in axes.ravel()[len(order) :]:
        ax.axis("off")
    if handles:
        fig.legend(
            handles[0],
            handles[1],
            frameon=False,
            loc="lower right",
            bbox_to_anchor=(0.96, 0.08),
        )
    fig.tight_layout()
    fig.savefig(OUT / "h1_wealth_13_markets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 4, figsize=(13.2, 10.8))
    for ax, market in zip(axes.ravel(), order):
        series = results[market]["series"]
        _, benchmark = best(series)
        ratio = (1 + series["strat"]).cumprod() / (1 + benchmark).cumprod()
        ax.axhline(1, color=GREY, lw=0.7)
        ax.plot(ratio.index.to_timestamp(), ratio, color=BLUE, lw=1.3)
        ax.fill_between(
            ratio.index.to_timestamp(),
            1,
            ratio.values,
            where=ratio.values >= 1,
            color=BLUE,
            alpha=0.14,
        )
        ax.fill_between(
            ratio.index.to_timestamp(),
            1,
            ratio.values,
            where=ratio.values < 1,
            color=ORANGE,
            alpha=0.12,
        )
        ax.text(
            0.03,
            0.95,
            rf"{R.NAMES[market]} \quad ${ratio.iloc[-1]:.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(order) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "h1_relative_wealth_13_markets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def era_figure(results: dict[str, dict]) -> None:
    groups = [
        ("fondateurs de l'euro", BLUE, ["DEU", "FRA", "ESP", "NLD", "BEL"]),
        ("flottants européens", ORANGE, ["CHE", "GBR", "NOR"]),
        ("hors Europe", PURPLE, ["USA", "CAN", "AUS", "JPN", "ZAF"]),
    ]
    rows = []
    for group, colour, markets in groups:
        for market in markets:
            series = results[market]["series"]
            index = series["strat"].index
            pre = index < CUT
            post = index >= CUT
            if pre.sum() < 60 or post.sum() < 60:
                continue

            def advantage(mask) -> float:
                bill = series["bill"][mask]
                strat = R.sharpe(series["strat"][mask], bill)
                b60 = R.sharpe(series["b6040"][mask], bill)
                pp = R.sharpe(series["pp"][mask], bill)
                return strat - max(b60, pp)

            rows.append(
                {
                    "market": market,
                    "group": group,
                    "colour": colour,
                    "pre": advantage(pre),
                    "post": advantage(post),
                }
            )
    positions = []
    current = len(rows) + len(groups) - 2
    fig, ax = plt.subplots(figsize=(9.8, 7.3))
    for group, colour, _ in groups:
        subset = sorted(
            [row for row in rows if row["group"] == group],
            key=lambda row: -row["pre"],
        )
        for row in subset:
            ax.plot(
                [row["pre"], row["post"]],
                [current, current],
                color=GRID,
                lw=2.4,
            )
            ax.scatter(row["pre"], current, s=68, color=colour, zorder=3)
            ax.scatter(
                row["post"],
                current,
                s=68,
                edgecolor=colour,
                facecolor=SURFACE,
                lw=1.6,
                zorder=3,
            )
            positions.append((current, R.NAMES[row["market"]]))
            current -= 1
        current -= 1
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks([position for position, _ in positions])
    ax.set_yticklabels([label for _, label in positions])
    ax.set_xlabel(r"avantage de Sharpe net sur la meilleure référence")
    handles = [
        Line2D([], [], marker="o", ls="", color=colour, label=group)
        for group, colour, _ in groups
    ]
    handles += [
        Line2D([], [], marker="o", ls="", color=INK, label="avant 1999"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=INK,
            markerfacecolor=SURFACE,
            label="après 1999",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(rect=(0, 0, 0.80, 1))
    fig.savefig(OUT / "era_1999_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def sensitivity() -> None:
    costs = pd.read_csv(OUT / "h1_cost_sensitivity.csv")
    windows = pd.read_csv(OUT / "h1_window_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    configurations = [
        (
            axes[0],
            costs["cost_oneway"].to_numpy() * 10000,
            costs,
            r"coût aller simple (points de base)",
            40,
        ),
        (
            axes[1],
            windows["window_months"].to_numpy(),
            windows,
            r"fenêtre du signal (mois)",
            84,
        ),
    ]
    for ax, x, data, xlabel, primary in configurations:
        ax.plot(
            x,
            data["median_advantage"],
            color=BLUE,
            marker="o",
            lw=1.7,
            label="avantage médian",
        )
        ax.axhline(0, color=INK, lw=0.7)
        ax.axvline(primary, color=GREY, ls=":", lw=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"avantage de Sharpe net médian", color=BLUE)
        ax.tick_params(axis="y", colors=BLUE)
        ax.grid(axis="y", color=GRID, lw=0.7)
        other = ax.twinx()
        other.plot(
            x,
            data["positive_markets"],
            color=GOLD,
            marker="s",
            ls="--",
            lw=1.3,
            label="marchés positifs",
        )
        other.set_ylim(0, 13.5)
        other.set_yticks([0, 4, 8, 12, 13])
        other.set_ylabel(r"nombre de marchés positifs", color=GOLD)
        other.tick_params(axis="y", colors=GOLD)
        for spine in ("top",):
            ax.spines[spine].set_visible(False)
            other.spines[spine].set_visible(False)
    fig.legend(
        handles=[
            Line2D([], [], color=BLUE, marker="o", label="avantage médian"),
            Line2D([], [], color=GOLD, marker="s", ls="--", label="marchés positifs"),
            Line2D([], [], color=GREY, ls=":", label="spécification principale"),
        ],
        frameon=False,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "h1_cost_window_sensitivity_13_markets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    setup()
    results = collect()
    signal_heatmap(results)
    hypothesis_ladder()
    jackknife()
    risk_figure()
    static_timing()
    wealth_figures(results)
    era_figure(results)
    sensitivity()
    print("wrote harmonised paper figure suite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
