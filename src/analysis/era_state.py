"""Audit H1 states and performance before and after January 2000.

Inputs: the frozen engine inputs and published full-sample H1 CSV files.
Outputs: monthly states, era summaries, rolling estimates, audit hashes, and
the four temporal figures in the complete-sample result directory.
Purpose: support the paper's descriptive, ex-post temporal analysis.

The split is descriptive and ex post. This script does not change the H1
portfolio, its market universe, or the primary inference.  It exports the
lagged state actually applied to portfolio weights, reconciles the existing
full-sample state shares and risk-adjusted panel, and produces the two French
figures used in the temporal-analysis section.
"""
from __future__ import annotations

import hashlib
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/four-quadrant-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from analysis import run_full_sample as R
from engine import engine as E
from figures.i18n import L, figpath, market_name
from figures.plot_style import (
    BLUE,
    GREY,
    GRID,
    INK,
    LIGHT_GREY,
    ORANGE,
    SURFACE,
    setup_matplotlib,
)

OUT = R.OUT
CUT = pd.Period("2000-01", freq="M")
DATE_TAG = "2026-07-27"

MONTHLY_NAME = f"h1_signal_states_monthly_{DATE_TAG}.csv"
SUMMARY_NAME = f"h1_signal_states_pre_post_2000_{DATE_TAG}.csv"
SYNC_MONTHLY_NAME = f"h1_signal_synchronization_monthly_{DATE_TAG}.csv"
SYNC_SUMMARY_NAME = f"h1_signal_synchronization_pre_post_2000_{DATE_TAG}.csv"
PERFORMANCE_NAME = f"h1_signal_state_performance_pre_post_2000_{DATE_TAG}.csv"
ROLLING_ALPHA_NAME = f"h1_rolling_alpha_60m_{DATE_TAG}.csv"
FIGURE_A_NAME = "h1_alpha_pre_post_2000_13_markets_fr.png"
FIGURE_B_NAME = "h1_signal_state_pre_post_2000_13_markets_fr.png"
FIGURE_C_NAME = "h1_signal_state_timeseries_13_markets_fr.png"
FIGURE_D_NAME = "h1_rolling_alpha_60m_13_markets_fr.png"
MANIFEST_NAME = f"h1_signal_state_outputs_sha256_{DATE_TAG}.csv"

def setup_plotting() -> None:
    setup_matplotlib()


def risk_adjusted_difference(
    strategy: pd.Series,
    reference: pd.Series,
    bill: pd.Series,
) -> pd.Series:
    strategy_excess = strategy - bill
    reference_excess = reference - bill
    return (
        strategy_excess
        / strategy_excess.rolling(36).std().shift(1)
        - reference_excess
        / reference_excess.rolling(36).std().shift(1)
    )


def collect_monthly() -> tuple[pd.DataFrame, dict[str, dict]]:
    R.wire_everything()
    E.run_market.return_series = True
    results: dict[str, dict] = {}
    parts: list[pd.DataFrame] = []

    for market in R.MARKETS:
        result = E.run_market(
            market,
            real_gate=False,
            bootstrap_draws=0,
        )
        results[market] = result
        series = result["series"]
        index = series["strat"].index
        state = (
            series["signal"]
            .reindex(index)
            .fillna(1.0)
            .astype(int)
        )
        benchmark_name, primary = R.harder_benchmark(series)

        frame = pd.DataFrame(index=index)
        frame["market"] = market
        frame["country"] = R.NAMES[market]
        frame["month"] = index.astype(str)
        frame["benchmark_primary"] = benchmark_name
        frame["bond_state"] = state
        frame["state"] = np.where(state.eq(1), "obligations", "or")
        frame["strategy_return"] = series["strat"].reindex(index)
        frame["bill_return"] = series["bill"].reindex(index)
        frame["primary_reference_return"] = primary.reindex(index)
        frame["permanent_quarterly_return"] = series["pp"].reindex(index)
        frame["permanent_monthly_return"] = series["pp_monthly"].reindex(index)
        frame["sixty_forty_return"] = series["b6040"].reindex(index)

        references = {
            "primary": primary,
            "permanent_quarterly": series["pp"],
            "permanent_monthly": series["pp_monthly"],
            "sixty_forty": series["b6040"],
        }
        for label, reference in references.items():
            frame[f"return_difference_{label}"] = (
                series["strat"] - reference
            ).reindex(index)
            frame[f"z_{label}"] = risk_adjusted_difference(
                series["strat"],
                reference,
                series["bill"],
            ).reindex(index)
        parts.append(frame.reset_index(drop=True))

    monthly = pd.concat(parts, ignore_index=True)
    monthly = monthly.sort_values(["market", "month"]).reset_index(drop=True)
    return monthly, results


def reconcile(monthly: pd.DataFrame) -> None:
    expected_market = pd.read_csv(OUT / "h1_per_market.csv").set_index("market")
    observed_share = monthly.groupby("market")["bond_state"].mean()
    for market in R.MARKETS:
        error = abs(
            float(observed_share.loc[market])
            - float(expected_market.loc[market, "bond_state_share"])
        )
        if error > 1e-12:
            raise RuntimeError(
                f"State-share reconciliation failed for {market}: {error}"
            )
        observed_benchmark = monthly.loc[
            monthly["market"].eq(market),
            "benchmark_primary",
        ].iloc[0]
        expected_benchmark = expected_market.loc[market, "benchmark"]
        if observed_benchmark != expected_benchmark:
            raise RuntimeError(
                f"Benchmark reconciliation failed for {market}: "
                f"{observed_benchmark} != {expected_benchmark}"
            )

    expected_panel = pd.read_csv(OUT / "h1_risk_adjusted_panel.csv")
    observed_panel = monthly.dropna(subset=["z_primary"])[
        ["market", "month", "z_primary"]
    ].rename(columns={"z_primary": "z_observed"})
    check = expected_panel.merge(
        observed_panel,
        on=["market", "month"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not check["_merge"].eq("both").all():
        counts = check["_merge"].value_counts().to_dict()
        raise RuntimeError(f"Risk-panel row reconciliation failed: {counts}")
    error = (check["z"] - check["z_observed"]).abs().max()
    if float(error) > 1e-12:
        raise RuntimeError(
            f"Risk-panel value reconciliation failed: {error}"
        )


def period_name(month: pd.Period) -> str:
    return "before_2000" if month < CUT else "from_2000"


def sequence_statistics(sample: pd.DataFrame) -> dict[str, float | int | str]:
    ordered = sample.sort_values("month").copy()
    periods = pd.PeriodIndex(ordered["month"], freq="M")
    states = ordered["bond_state"].astype(int).reset_index(drop=True)
    gaps = pd.Series(False, index=states.index)
    if len(periods) > 1:
        gaps.iloc[1:] = [
            (periods[i] - periods[i - 1]).n != 1
            for i in range(1, len(periods))
        ]
    previous_exists = states.shift(1).notna()
    comparable = previous_exists & ~gaps
    transitions = int(
        ((states != states.shift(1)) & comparable).sum()
    )
    comparable_intervals = int(comparable.sum())
    annualized = (
        transitions / (comparable_intervals / 12)
        if comparable_intervals
        else np.nan
    )
    new_sequence = (
        ~previous_exists
        | gaps
        | (states != states.shift(1))
    )
    run_id = new_sequence.cumsum()
    durations = states.groupby(run_id).size().astype(float)
    return {
        "observations": len(ordered),
        "start": ordered["month"].min(),
        "end": ordered["month"].max(),
        "bond_state_share": float(states.mean()),
        "gold_state_share": float(1 - states.mean()),
        "transitions": transitions,
        "comparable_month_pairs": comparable_intervals,
        "transitions_annualized": float(annualized),
        "sequences": int(len(durations)),
        "mean_sequence_months": float(durations.mean()),
        "median_sequence_months": float(durations.median()),
        "max_sequence_months": int(durations.max()),
    }


def state_summaries(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = [
        ("full_strategy", monthly),
        ("risk_adjusted", monthly.dropna(subset=["z_primary"])),
    ]
    for sample_name, base in samples:
        for market, market_data in base.groupby("market", sort=True):
            market_data = market_data.copy()
            market_data["period"] = pd.PeriodIndex(
                market_data["month"], freq="M"
            ).map(period_name)
            for period, period_data in market_data.groupby("period", sort=True):
                rows.append(
                    {
                        "sample": sample_name,
                        "market": market,
                        "country": R.NAMES[market],
                        "period": period,
                        **sequence_statistics(period_data),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["sample", "market", "period"]
    )


def synchronization(
    monthly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk = monthly.dropna(subset=["z_primary"]).copy()
    risk["period"] = pd.PeriodIndex(risk["month"], freq="M").map(period_name)
    rows: list[dict[str, object]] = []
    for (period, month), data in risk.groupby(["period", "month"], sort=True):
        observed = len(data)
        bonds = int(data["bond_state"].sum())
        gold = observed - bonds
        majority = max(bonds, gold) / observed
        if observed >= 2:
            agreeing_pairs = bonds * (bonds - 1) + gold * (gold - 1)
            total_ordered_pairs = observed * (observed - 1)
            pairwise = agreeing_pairs / total_ordered_pairs
        else:
            pairwise = np.nan
        rows.append(
            {
                "period": period,
                "month": month,
                "markets_observed": observed,
                "bond_state_share": bonds / observed,
                "majority_state_share": majority,
                "pairwise_state_agreement": pairwise,
                "unanimous_state": bonds in (0, observed),
            }
        )
    monthly_sync = pd.DataFrame(rows)
    summary = (
        monthly_sync.groupby("period", sort=True)
        .agg(
            months=("month", "size"),
            mean_markets_observed=("markets_observed", "mean"),
            mean_bond_state_share=("bond_state_share", "mean"),
            mean_majority_state_share=("majority_state_share", "mean"),
            mean_pairwise_state_agreement=(
                "pairwise_state_agreement",
                "mean",
            ),
            unanimous_month_share=("unanimous_state", "mean"),
        )
        .reset_index()
    )
    return monthly_sync, summary


def hac_mean(values: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float)
    if clean.empty:
        return {
            "observations": 0,
            "mean": np.nan,
            "standard_error_hac12": np.nan,
            "t_statistic": np.nan,
            "p_one_sided_normal": np.nan,
            "p_two_sided_normal": np.nan,
        }
    if len(clean) == 1:
        return {
            "observations": 1,
            "mean": float(clean.iloc[0]),
            "standard_error_hac12": np.nan,
            "t_statistic": np.nan,
            "p_one_sided_normal": np.nan,
            "p_two_sided_normal": np.nan,
        }
    maxlags = min(12, max(0, len(clean) - 1))
    fit = sm.OLS(
        clean.to_numpy(),
        np.ones((len(clean), 1)),
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": maxlags, "use_correction": True},
    )
    t_statistic = float(fit.tvalues[0])
    return {
        "observations": len(clean),
        "mean": float(fit.params[0]),
        "standard_error_hac12": float(fit.bse[0]),
        "t_statistic": t_statistic,
        "p_one_sided_normal": float(norm.sf(t_statistic)),
        "p_two_sided_normal": float(2 * norm.sf(abs(t_statistic))),
    }


def performance_summaries(monthly: pd.DataFrame) -> pd.DataFrame:
    data = monthly.copy()
    data["period"] = pd.PeriodIndex(data["month"], freq="M").map(period_name)
    references = [
        ("primary", "z_primary", "return_difference_primary"),
        (
            "permanent_quarterly",
            "z_permanent_quarterly",
            "return_difference_permanent_quarterly",
        ),
        (
            "permanent_monthly",
            "z_permanent_monthly",
            "return_difference_permanent_monthly",
        ),
        ("sixty_forty", "z_sixty_forty", "return_difference_sixty_forty"),
    ]
    rows: list[dict[str, object]] = []
    for market, market_data in data.groupby("market", sort=True):
        for period, period_data in market_data.groupby("period", sort=True):
            states: list[tuple[str, pd.DataFrame]] = [
                ("all", period_data),
                (
                    "obligations",
                    period_data[period_data["bond_state"].eq(1)],
                ),
                ("or", period_data[period_data["bond_state"].eq(0)]),
            ]
            for reference, z_column, return_column in references:
                for state, state_data in states:
                    z_stats = hac_mean(state_data[z_column])
                    raw_stats = hac_mean(state_data[return_column])
                    rows.append(
                        {
                            "market": market,
                            "country": R.NAMES[market],
                            "period": period,
                            "state": state,
                            "reference": reference,
                            "observations": z_stats["observations"],
                            "alpha_risk_adjusted_annualized": (
                                float(z_stats["mean"]) * np.sqrt(12)
                            ),
                            "standard_error_hac12_annualized": (
                                float(z_stats["standard_error_hac12"])
                                * np.sqrt(12)
                            ),
                            "t_statistic": z_stats["t_statistic"],
                            "p_one_sided_normal": z_stats[
                                "p_one_sided_normal"
                            ],
                            "p_two_sided_normal": z_stats[
                                "p_two_sided_normal"
                            ],
                            "raw_return_observations": raw_stats[
                                "observations"
                            ],
                            "raw_return_difference_annualized": (
                                float(raw_stats["mean"]) * 12
                            ),
                            "raw_return_standard_error_hac12_annualized": (
                                float(raw_stats["standard_error_hac12"]) * 12
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def load_alpha_data() -> pd.DataFrame:
    data = pd.read_csv(
        OUT / f"post_2000_h1_market_means_{DATE_TAG}.csv"
    )
    data["ci95_low"] = (
        data["alpha_annualized_sharpe"]
        - 1.96 * data["hac12_standard_error_annualized"]
    )
    data["ci95_high"] = (
        data["alpha_annualized_sharpe"]
        + 1.96 * data["hac12_standard_error_annualized"]
    )
    return data


def reconcile_derived_outputs(
    summary: pd.DataFrame,
    performance: pd.DataFrame,
    alpha: pd.DataFrame,
) -> None:
    risk = summary[summary["sample"].eq("risk_adjusted")]
    expected_period_observations = {
        "before_2000": 1807,
        "from_2000": 3989,
    }
    observed = risk.groupby("period")["observations"].sum().to_dict()
    if observed != expected_period_observations:
        raise RuntimeError(
            f"State-summary observation reconciliation failed: {observed}"
        )

    primary = performance[
        performance["state"].eq("all")
        & performance["reference"].eq("primary")
    ][
        [
            "market",
            "period",
            "observations",
            "alpha_risk_adjusted_annualized",
        ]
    ]
    check = alpha.merge(
        primary,
        on=["market", "period"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not check["_merge"].eq("both").all():
        raise RuntimeError("Performance-summary row reconciliation failed")
    if not (
        check["observations_x"].astype(int)
        == check["observations_y"].astype(int)
    ).all():
        raise RuntimeError(
            "Performance-summary observation reconciliation failed"
        )
    error = (
        check["alpha_annualized_sharpe"]
        - check["alpha_risk_adjusted_annualized"]
    ).abs().max()
    if float(error) > 1e-12:
        raise RuntimeError(
            f"Performance-summary alpha reconciliation failed: {error}"
        )


def figure_a(alpha: pd.DataFrame) -> None:
    pivot = alpha.pivot(index="market", columns="period")
    pivot["change"] = (
        pivot[("alpha_annualized_sharpe", "from_2000")]
        - pivot[("alpha_annualized_sharpe", "before_2000")]
    )
    order = pivot["change"].sort_values().index.tolist()
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(9.7, 7.2))

    for position, market in zip(y, order):
        pre = float(
            pivot.loc[market, ("alpha_annualized_sharpe", "before_2000")]
        )
        post = float(
            pivot.loc[market, ("alpha_annualized_sharpe", "from_2000")]
        )
        pre_se = float(
            pivot.loc[
                market,
                ("hac12_standard_error_annualized", "before_2000"),
            ]
        )
        post_se = float(
            pivot.loc[
                market,
                ("hac12_standard_error_annualized", "from_2000"),
            ]
        )
        pre_n = int(pivot.loc[market, ("observations", "before_2000")])
        colour = ORANGE if market in {"USA", "CAN"} else BLUE
        if pre_n < 60:
            colour = GREY
        ax.plot([pre, post], [position, position], color=GRID, lw=2.5, zorder=1)
        ax.errorbar(
            pre,
            position,
            xerr=1.96 * pre_se,
            fmt="o",
            ms=6.4,
            color=colour,
            ecolor=colour,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.9,
            zorder=3,
        )
        ax.errorbar(
            post,
            position,
            xerr=1.96 * post_se,
            fmt="o",
            ms=6.4,
            markerfacecolor=SURFACE,
            markeredgecolor=colour,
            markeredgewidth=1.4,
            ecolor=colour,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.9,
            zorder=3,
        )

    labels = [
        market_name(market, R.NAMES[market])
        + (r"$^{\dagger}$" if int(
            pivot.loc[market, ("observations", "before_2000")]
        ) < 60 else "")
        for market in order
    ]
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(
        L(
            r"avantage H1 ajusté du risque (ratio de Sharpe annualisé)",
            r"H1 risk-adjusted advantage (annualized Sharpe ratio)",
        )
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="", color=BLUE,
                label=L(r"avant janvier 2000", r"before January 2000"),
            ),
            Line2D(
                [], [], marker="o", ls="", color=BLUE,
                markerfacecolor=SURFACE,
                label=L(r"depuis janvier 2000", r"since January 2000"),
            ),
            Line2D(
                [], [], color=ORANGE, lw=2,
                label=L(r"États-Unis et Canada", r"United States and Canada"),
            ),
            Line2D(
                [], [], color=GREY, lw=2,
                label=L(
                    r"$\dagger$ moins de 60 mois avant 2000",
                    r"$\dagger$ fewer than 60 months before 2000",
                ),
            ),
        ],
        frameon=False,
        loc="lower right",
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(figpath(OUT / FIGURE_A_NAME), dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_b(summary: pd.DataFrame) -> None:
    # Figure B describes the state itself, not the risk-adjusted alpha.  Use
    # every month in which the implemented portfolio state exists.  The
    # risk-adjusted sample starts 36 months later because it needs a trailing
    # volatility estimate and would therefore hide genuine early transitions.
    data = summary[summary["sample"].eq("full_strategy")].copy()
    shares = data.pivot(index="market", columns="period")
    order = (
        shares[("bond_state_share", "before_2000")]
        - shares[("bond_state_share", "from_2000")]
    ).abs().sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 7.2),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    specifications = [
        (
            axes[0],
            "bond_state_share",
            L(
                r"part des mois en état obligataire",
                r"share of months in the bond state",
            ),
            (-0.03, 1.03),
        ),
        (
            axes[1],
            "transitions_annualized",
            L(r"transitions d'état par an", r"state transitions per year"),
            None,
        ),
    ]
    for ax, variable, xlabel, limits in specifications:
        for position, market in zip(y, order):
            pre = float(shares.loc[market, (variable, "before_2000")])
            post = float(shares.loc[market, (variable, "from_2000")])
            ax.plot([pre, post], [position, position], color=GRID, lw=2.5)
            ax.scatter(pre, position, s=54, color=BLUE, zorder=3)
            ax.scatter(
                post,
                position,
                s=54,
                facecolor=SURFACE,
                edgecolor=ORANGE,
                linewidth=1.5,
                zorder=3,
            )
        if limits is not None:
            ax.set_xlim(*limits)
        else:
            ax.margins(x=0.06)
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, color=GRID, lw=0.7)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)

    labels = [
        market_name(market, R.NAMES[market])
        + (
            r"$^{\dagger}$"
            if int(shares.loc[market, ("observations", "before_2000")]) < 60
            else ""
        )
        for market in order
    ]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].legend(
        handles=[
            Line2D(
                [], [], marker="o", ls="", color=BLUE,
                label=L(r"avant janvier 2000", r"before January 2000"),
            ),
            Line2D(
                [], [], marker="o", ls="", color=ORANGE,
                markerfacecolor=SURFACE,
                label=L(r"depuis janvier 2000", r"since January 2000"),
            ),
            Line2D(
                [], [], color=GREY, lw=2,
                label=L(
                    r"$\dagger$ moins de 60 mois",
                    r"$\dagger$ fewer than 60 months",
                ),
            ),
        ],
        frameon=False,
        loc="lower right",
    )
    fig.subplots_adjust(
        left=0.10,
        right=0.99,
        bottom=0.12,
        top=0.99,
        wspace=0.10,
    )
    fig.savefig(figpath(OUT / FIGURE_B_NAME), dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_c(monthly: pd.DataFrame) -> None:
    """Plot the complete implemented state history as one row per market."""
    data = monthly[["market", "country", "month", "bond_state"]].copy()
    data["period"] = pd.PeriodIndex(data["month"], freq="M")
    order = (
        pd.read_csv(OUT / "h1_per_market.csv")
        .sort_values("advantage", ascending=False)["market"]
        .tolist()
    )
    periods = pd.period_range(
        data["period"].min(),
        data["period"].max(),
        freq="M",
    )
    matrix = (
        data.pivot(index="market", columns="period", values="bond_state")
        .reindex(index=order, columns=periods)
    )

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
    ax.axvline(
        cut_date,
        color=INK,
        lw=1.1,
        ls=(0, (4, 3)),
        zorder=3,
    )
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([market_name(market, R.NAMES[market]) for market in order])
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
            Patch(
                facecolor=BLUE, edgecolor="none",
                label=L(r"état obligations", r"bond state"),
            ),
            Patch(
                facecolor=ORANGE, edgecolor="none",
                label=L(r"état or", r"gold state"),
            ),
            Line2D(
                [], [], color=INK, lw=1.1, ls=(0, (4, 3)),
                label=L(r"janvier 2000", r"January 2000"),
            ),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.99,
        bottom=0.20,
        top=0.99,
    )
    fig.savefig(figpath(OUT / FIGURE_C_NAME), dpi=220, bbox_inches="tight")
    plt.close(fig)


def rolling_alpha(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return the paper's risk-adjusted alpha in trailing 60-month windows."""
    data = monthly[
        ["market", "country", "month", "z_primary"]
    ].copy()
    data = data.sort_values(["market", "month"]).reset_index(drop=True)
    data["alpha_rolling_60m"] = (
        data.groupby("market", sort=False)["z_primary"]
        .transform(
            lambda values: (
                values.rolling(60, min_periods=60).mean() * np.sqrt(12)
            )
        )
    )
    data["markets_available"] = np.where(
        data["alpha_rolling_60m"].notna(),
        1,
        0,
    )

    aggregate = (
        data.groupby("month", as_index=False)
        .agg(
            alpha_rolling_60m=("alpha_rolling_60m", "mean"),
            markets_available=("markets_available", "sum"),
        )
    )
    # Avoid calling a one- or two-country series an international aggregate.
    aggregate.loc[
        aggregate["markets_available"] < 8,
        "alpha_rolling_60m",
    ] = np.nan
    aggregate["market"] = "EQUAL_WEIGHT"
    aggregate["country"] = "Moyenne équipondérée"
    aggregate["z_primary"] = np.nan
    aggregate = aggregate[data.columns]
    return pd.concat([aggregate, data], ignore_index=True)


def figure_d(rolling: pd.DataFrame) -> None:
    """Plot rolling risk-adjusted alpha for the aggregate and every market."""
    market_order = (
        rolling[
            rolling["market"].ne("EQUAL_WEIGHT")
            & rolling["alpha_rolling_60m"].notna()
        ]
        .groupby(["market", "country"], as_index=False)["month"]
        .min()
        .sort_values(["month", "country"])["market"]
        .tolist()
    )
    panels = ["EQUAL_WEIGHT", *market_order]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(8.2, 11.4),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": 0.08},
    )
    x_min = pd.Timestamp("1968-01-01")
    x_max = pd.Timestamp("2027-01-01")
    cut_date = CUT.to_timestamp(how="start")

    for index, (ax, market) in enumerate(zip(axes, panels)):
        panel = rolling[
            rolling["market"].eq(market)
            & rolling["alpha_rolling_60m"].notna()
        ].copy()
        dates = pd.to_datetime(panel["month"])
        is_aggregate = market == "EQUAL_WEIGHT"
        color = BLUE if is_aggregate else INK
        linewidth = 1.8 if is_aggregate else 1.0
        ax.plot(
            dates,
            panel["alpha_rolling_60m"],
            color=color,
            lw=linewidth,
        )
        ax.axhline(0, color=GREY, lw=0.7, zorder=0)
        ax.axvline(
            cut_date,
            color=ORANGE,
            lw=0.8,
            ls=(0, (4, 3)),
            zorder=0,
        )
        label = (
            L(r"Moyenne équipondérée", r"Equal-weighted average")
            if is_aggregate
            else market_name(market, R.NAMES[market])
        )
        ax.text(
            0.01,
            0.82,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.6,
            color=color,
        )
        ax.set_ylim(-1.10, 1.52)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(axis="y", labelsize=7.2, length=2)
        ax.yaxis.grid(False)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if index < len(panels) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", which="both", length=0)

    axes[-1].set_xlim(x_min, x_max)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(10))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel(L(r"mois", r"month"))
    fig.supylabel(
        L(
            r"alpha H1 mobile, ratio de Sharpe annualisé",
            r"rolling H1 alpha, annualized Sharpe ratio",
        ),
        x=0.005,
        fontsize=10,
    )
    axes[-1].legend(
        handles=[
            Line2D(
                [], [], color=BLUE, lw=1.8,
                label=L(r"moyenne équipondérée", r"equal-weighted average"),
            ),
            Line2D(
                [], [], color=INK, lw=1.0,
                label=L(r"pays", r"country"),
            ),
            Line2D(
                [], [], color=ORANGE, lw=0.8, ls=(0, (4, 3)),
                label=L(r"janvier 2000", r"January 2000"),
            ),
        ],
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.70),
        fontsize=8.2,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        bottom=0.09,
        top=0.995,
    )
    fig.savefig(figpath(OUT / FIGURE_D_NAME), dpi=220, bbox_inches="tight")
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: list[Path]) -> None:
    rows = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    pd.DataFrame(rows).to_csv(OUT / MANIFEST_NAME, index=False)


def main() -> int:
    setup_plotting()
    monthly, _ = collect_monthly()
    reconcile(monthly)

    summary = state_summaries(monthly)
    sync_monthly, sync_summary = synchronization(monthly)
    performance = performance_summaries(monthly)
    rolling = rolling_alpha(monthly)
    alpha = load_alpha_data()
    reconcile_derived_outputs(summary, performance, alpha)

    output_paths = [
        OUT / MONTHLY_NAME,
        OUT / SUMMARY_NAME,
        OUT / SYNC_MONTHLY_NAME,
        OUT / SYNC_SUMMARY_NAME,
        OUT / PERFORMANCE_NAME,
        OUT / ROLLING_ALPHA_NAME,
        OUT / FIGURE_A_NAME,
        OUT / FIGURE_B_NAME,
        OUT / FIGURE_C_NAME,
        OUT / FIGURE_D_NAME,
    ]
    monthly.to_csv(output_paths[0], index=False)
    summary.to_csv(output_paths[1], index=False)
    sync_monthly.to_csv(output_paths[2], index=False)
    sync_summary.to_csv(output_paths[3], index=False)
    performance.to_csv(output_paths[4], index=False)
    rolling.to_csv(output_paths[5], index=False)
    figure_a(alpha)
    figure_b(summary)
    figure_c(monthly)
    figure_d(rolling)
    write_manifest(output_paths)

    reported = summary[
        summary["sample"].isin(["full_strategy", "risk_adjusted"])
    ]
    print("STATE SHARE RECONCILIATION: PASS")
    print("RISK-ADJUSTED PANEL RECONCILIATION: PASS")
    print("DERIVED OUTPUT RECONCILIATION: PASS")
    print(
        reported.groupby(["sample", "period"])
        .agg(
            countries=("market", "nunique"),
            observations=("observations", "sum"),
            countries_without_transition=(
                "transitions",
                lambda values: int((values == 0).sum()),
            ),
            total_transitions=("transitions", "sum"),
            mean_bond_share=("bond_state_share", "mean"),
            median_bond_share=("bond_state_share", "median"),
            median_transitions_per_year=(
                "transitions_annualized",
                "median",
            ),
            median_sequence_months=("median_sequence_months", "median"),
        )
        .to_string()
    )
    print(sync_summary.to_string(index=False))
    print(f"wrote {len(output_paths)} outputs and {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
