"""Date H1's Sharpe advantage against an exposure-matched static portfolio.

Inputs: the frozen engine inputs and published H1 market summary.
Outputs: country, summary, monthly, inference, contrast, and hash-manifest CSV
files in the complete-sample result directory.
Purpose: separate descriptive era timing from average bond/gold exposure.

Test A splits the already published dynamic and full-history matched-static
return series before and after January 2000.  It answers when the published
median Sharpe difference of +0.147 was earned.

Test B rebuilds both portfolios at each era boundary.  Its static comparator
uses the era-specific mean bond/gold exposure, first with quarterly and then
with monthly rebalancing.  It answers whether timing adds value within an era
at equal average bond/gold exposure.

Direct Sharpe differences receive paired circular-block bootstrap intervals.
Monthly risk-normalized differences are separate estimands and receive
calendar-month clustered and Driscoll-Kraay covariance estimates.  A joint
regression with a post-2000 indicator tests the direct difference between
the two era-specific normalized timing estimates.
"""
from __future__ import annotations
from pathlib import Path

import hashlib

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import binomtest, norm

from analysis import run_full_sample as R
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

OUT = FULL_SAMPLE_RESULTS

CUT = pd.Period("2000-01", freq="M")
BLOCK_MONTHS = 12
BOOTSTRAP_DRAWS = 9999
SEED = 20260727

COUNTRY_NAME = "h1_timing_era_country.csv"
SUMMARY_NAME = "h1_timing_era_summary.csv"
MONTHLY_NAME = "h1_timing_era_monthly_z.csv"
INFERENCE_NAME = "h1_timing_era_inference.csv"
CONTRAST_NAME = "h1_timing_era_contrast.csv"
MANIFEST_NAME = "h1_timing_era_sha256.csv"


def sharpe(excess_return: pd.Series | np.ndarray) -> float:
    values = np.asarray(excess_return, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    return float(values.mean() / values.std(ddof=1) * np.sqrt(12))


def period_name(month: pd.Period) -> str:
    return "before_2000" if month < CUT else "from_2000"


def portfolio(
    assets: pd.DataFrame,
    target,
    rebalance: str,
    cost_oneway: float = E.COST_ONEWAY,
) -> pd.Series:
    """Replicate h1_engine.portfolio on a supplied evaluation period."""
    previous_weights: dict[str, float] | None = None
    previous_month: pd.Period | None = None
    output: dict[pd.Period, float] = {}
    for month in assets.index:
        target_weights = target(month)
        if previous_weights is None:
            weights = target_weights
            cost = cost_oneway * sum(abs(value) for value in weights.values())
        else:
            assert previous_month is not None
            drift = {
                asset: previous_weights[asset]
                * (1 + assets.loc[previous_month, asset])
                for asset in previous_weights
            }
            total = sum(drift.values())
            drift = {asset: value / total for asset, value in drift.items()}
            unchanged = target(month) == target(previous_month)
            if (
                rebalance == "Q"
                and month.month not in (3, 6, 9, 12)
                and unchanged
            ):
                weights = drift
                cost = 0.0
            else:
                weights = target_weights
                cost = cost_oneway * sum(
                    abs(weights[asset] - drift.get(asset, 0.0))
                    for asset in weights
                )
        output[month] = (
            sum(
                weights[asset] * assets.loc[month, asset]
                for asset in weights
            )
            - cost
        )
        previous_weights = weights
        previous_month = month
    return pd.Series(output, dtype=float)


def dynamic_target(signal: pd.Series):
    def target(month: pd.Period) -> dict[str, float]:
        state = int(signal.loc[month])
        fourth = "bond" if state else "gold"
        weights = {"eq": 0.25, "bond": 0.25, "bill": 0.25, "gold": 0.0}
        weights[fourth] = weights.get(fourth, 0.0) + 0.25
        return weights

    return target


def static_target(bond_state_share: float):
    weights = {
        "eq": 0.25,
        "bill": 0.25,
        "bond": 0.25 + 0.25 * bond_state_share,
        "gold": 0.25 * (1 - bond_state_share),
    }
    return lambda month: weights.copy()


def circular_block_interval(
    dynamic_excess: pd.Series,
    static_excess: pd.Series,
    seed: int,
) -> tuple[float, float, float]:
    aligned = pd.concat(
        [dynamic_excess.rename("dynamic"), static_excess.rename("static")],
        axis=1,
    ).dropna()
    dynamic = aligned["dynamic"].to_numpy(float)
    static = aligned["static"].to_numpy(float)
    n = len(aligned)
    if n < 2:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    blocks = (n + BLOCK_MONTHS - 1) // BLOCK_MONTHS
    draws = np.empty(BOOTSTRAP_DRAWS)
    done = 0
    while done < BOOTSTRAP_DRAWS:
        batch = min(500, BOOTSTRAP_DRAWS - done)
        starts = rng.integers(0, n, size=(batch, blocks))
        offsets = np.arange(BLOCK_MONTHS)
        indices = (
            starts[:, :, None] + offsets[None, None, :]
        ) % n
        indices = indices.reshape(batch, -1)[:, :n]
        sampled_dynamic = dynamic[indices]
        sampled_static = static[indices]
        dynamic_sharpe = (
            sampled_dynamic.mean(axis=1)
            / sampled_dynamic.std(axis=1, ddof=1)
            * np.sqrt(12)
        )
        static_sharpe = (
            sampled_static.mean(axis=1)
            / sampled_static.std(axis=1, ddof=1)
            * np.sqrt(12)
        )
        draws[done : done + batch] = dynamic_sharpe - static_sharpe
        done += batch
    return (
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        float(np.mean(draws <= 0)),
    )


def risk_normalized_difference(
    dynamic: pd.Series,
    static: pd.Series,
    bill: pd.Series,
) -> pd.Series:
    dynamic_excess = dynamic - bill
    static_excess = static - bill
    return (
        dynamic_excess
        / dynamic_excess.rolling(36).std().shift(1)
        - static_excess
        / static_excess.rolling(36).std().shift(1)
    )


def country_row(
    market: str,
    test: str,
    comparator: str,
    period: str,
    dynamic: pd.Series,
    static: pd.Series,
    bill: pd.Series,
    bond_state_share: float,
    seed: int,
) -> dict[str, object]:
    aligned = pd.concat(
        [
            dynamic.rename("dynamic"),
            static.rename("static"),
            bill.rename("bill"),
        ],
        axis=1,
    ).dropna()
    dynamic_excess = aligned["dynamic"] - aligned["bill"]
    static_excess = aligned["static"] - aligned["bill"]
    dynamic_sharpe = sharpe(dynamic_excess)
    static_sharpe = sharpe(static_excess)
    low, high, probability_nonpositive = circular_block_interval(
        dynamic_excess,
        static_excess,
        seed,
    )
    return {
        "test": test,
        "comparator": comparator,
        "period": period,
        "market": market,
        "country": R.NAMES[market],
        "observations": len(aligned),
        "start": str(aligned.index.min()),
        "end": str(aligned.index.max()),
        "bond_state_share_period": bond_state_share,
        "dynamic_sharpe": dynamic_sharpe,
        "static_sharpe": static_sharpe,
        "delta_sharpe": dynamic_sharpe - static_sharpe,
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "bootstrap_probability_nonpositive": probability_nonpositive,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_block_months": BLOCK_MONTHS,
    }


def reconcile_full_history(
    market: str,
    series: dict[str, pd.Series],
) -> None:
    assets = pd.DataFrame(
        {
            "eq": series["eq"],
            "bond": series["bond"],
            "gold": series["gold"],
            "bill": series["bill"],
        }
    )
    signal = series["signal"].reindex(assets.index).fillna(1).astype(int)
    reconstructed_dynamic = portfolio(
        assets,
        dynamic_target(signal),
        "M",
    )
    share = float(signal.mean())
    reconstructed_static = portfolio(
        assets,
        static_target(share),
        "Q",
    )
    dynamic_error = float(
        (reconstructed_dynamic - series["strat"]).abs().max()
    )
    static_error = float(
        (reconstructed_static - series["matched"]).abs().max()
    )
    if dynamic_error > 1e-12 or static_error > 1e-12:
        raise RuntimeError(
            f"Portfolio reconstruction failed for {market}: "
            f"dynamic={dynamic_error}, static={static_error}"
        )


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    R.wire_everything()
    E.run_market.return_series = True
    published = pd.read_csv(OUT / "h1_per_market.csv").set_index("market")
    country_rows: list[dict[str, object]] = []
    monthly_rows: list[pd.DataFrame] = []

    for market_index, market in enumerate(R.MARKETS):
        result = E.run_market(
            market,
            real_gate=False,
            bootstrap_draws=0,
        )
        series = result["series"]
        reconcile_full_history(market, series)
        full_delta = (
            sharpe(series["strat"] - series["bill"])
            - sharpe(series["matched"] - series["bill"])
        )
        expected = float(published.loc[market, "advantage_vs_matched"])
        if abs(full_delta - expected) > 1e-12:
            raise RuntimeError(
                f"Published +0.147 reconciliation failed for {market}: "
                f"{full_delta} != {expected}"
            )

        full_z_a = risk_normalized_difference(
            series["strat"],
            series["matched"],
            series["bill"],
        )
        months = series["strat"].index
        signal = (
            series["signal"]
            .reindex(months)
            .fillna(1)
            .astype(int)
        )
        assets = pd.DataFrame(
            {
                "eq": series["eq"],
                "bond": series["bond"],
                "gold": series["gold"],
                "bill": series["bill"],
            }
        ).reindex(months)

        for period_index, (period, mask) in enumerate(
            [
                ("before_2000", months < CUT),
                ("from_2000", months >= CUT),
            ]
        ):
            period_months = months[mask]
            period_signal = signal.reindex(period_months)
            period_assets = assets.reindex(period_months)
            period_share = float(period_signal.mean())
            seed_base = SEED + market_index * 100 + period_index * 10

            dynamic_a = series["strat"].reindex(period_months)
            static_a = series["matched"].reindex(period_months)
            bill_a = series["bill"].reindex(period_months)
            country_rows.append(
                country_row(
                    market,
                    "A",
                    "full_history_static_quarterly",
                    period,
                    dynamic_a,
                    static_a,
                    bill_a,
                    period_share,
                    seed_base + 1,
                )
            )
            z_a = full_z_a.reindex(period_months)
            monthly_rows.append(
                pd.DataFrame(
                    {
                        "test": "A",
                        "comparator": "full_history_static_quarterly",
                        "period": period,
                        "market": market,
                        "country": R.NAMES[market],
                        "month": period_months.astype(str),
                        "z": z_a.to_numpy(),
                    }
                )
            )

            dynamic_b = portfolio(
                period_assets,
                dynamic_target(period_signal),
                "M",
            )
            for comparator_index, frequency in enumerate(["Q", "M"]):
                comparator = (
                    "era_static_quarterly"
                    if frequency == "Q"
                    else "era_static_monthly"
                )
                static_b = portfolio(
                    period_assets,
                    static_target(period_share),
                    frequency,
                )
                if (
                    frequency == "M"
                    and period_signal.nunique() == 1
                    and float((dynamic_b - static_b).abs().max()) > 1e-12
                ):
                    raise RuntimeError(
                        f"Constant-state monthly comparator failed for "
                        f"{market}/{period}"
                    )
                country_rows.append(
                    country_row(
                        market,
                        "B",
                        comparator,
                        period,
                        dynamic_b,
                        static_b,
                        period_assets["bill"],
                        period_share,
                        seed_base + 2 + comparator_index,
                    )
                )
                z_b = risk_normalized_difference(
                    dynamic_b,
                    static_b,
                    period_assets["bill"],
                )
                monthly_rows.append(
                    pd.DataFrame(
                        {
                            "test": "B",
                            "comparator": comparator,
                            "period": period,
                            "market": market,
                            "country": R.NAMES[market],
                            "month": period_months.astype(str),
                            "z": z_b.to_numpy(),
                        }
                    )
                )

    countries = pd.DataFrame(country_rows)
    expected_median = float(published["advantage_vs_matched"].median())
    expected_mean = float(published["advantage_vs_matched"].mean())
    if abs(expected_median - 0.147) > 0.001:
        raise RuntimeError(f"Unexpected published median: {expected_median}")
    full_reconstructed = []
    for market in R.MARKETS:
        rows = countries[
            countries["market"].eq(market)
            & countries["test"].eq("A")
        ]
        # The split is not additive in Sharpe.  Full-history reconciliation was
        # already performed above; this guard simply ensures two era rows exist.
        if len(rows) != 2:
            raise RuntimeError(f"Missing Test A era for {market}")
        full_reconstructed.append(market)
    if len(full_reconstructed) != 13 or abs(expected_mean - 0.113) > 0.001:
        raise RuntimeError("Published static-timing summary reconciliation failed")
    monthly = pd.concat(monthly_rows, ignore_index=True)
    return countries, monthly


def summarize_countries(countries: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["test", "comparator", "period"]
    for key, sample in countries.groupby(keys, sort=True):
        for scope, scoped in [
            ("all_available", sample),
            (
                "minimum_60_months",
                sample[sample["observations"].ge(60)],
            ),
        ]:
            positives = int(scoped["delta_sharpe"].gt(1e-12).sum())
            count = len(scoped)
            sign_p = (
                float(
                    binomtest(
                        positives,
                        count,
                        p=0.5,
                        alternative="greater",
                    ).pvalue
                )
                if count
                else np.nan
            )
            rows.append(
                {
                    "test": key[0],
                    "comparator": key[1],
                    "period": key[2],
                    "scope": scope,
                    "markets": count,
                    "observations": int(scoped["observations"].sum()),
                    "positive_markets": positives,
                    "mean_delta_sharpe": float(
                        scoped["delta_sharpe"].mean()
                    ),
                    "median_delta_sharpe": float(
                        scoped["delta_sharpe"].median()
                    ),
                    "minimum_delta_sharpe": float(
                        scoped["delta_sharpe"].min()
                    ),
                    "maximum_delta_sharpe": float(
                        scoped["delta_sharpe"].max()
                    ),
                    "sign_test_p_one_sided": sign_p,
                }
            )
    return pd.DataFrame(rows)


def panel_inference(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["test", "comparator", "period"]
    for key, base in monthly.dropna(subset=["z"]).groupby(keys, sort=True):
        counts = base.groupby("market")["z"].size()
        for scope, eligible in [
            ("all_available", counts.index),
            ("minimum_60_months", counts[counts.ge(60)].index),
        ]:
            sample = base[base["market"].isin(eligible)].copy()
            sample["time"] = pd.PeriodIndex(
                sample["month"],
                freq="M",
            ).to_timestamp()
            indexed = sample.set_index(["market", "time"]).sort_index()
            exog = pd.DataFrame({"constant": 1.0}, index=indexed.index)
            model = PanelOLS(indexed["z"].astype(float), exog)
            covariance_specs = [
                (
                    "calendar_month_cluster",
                    {"cov_type": "clustered", "cluster_time": True},
                ),
                (
                    "driscoll_kraay_12",
                    {
                        "cov_type": "kernel",
                        "kernel": "bartlett",
                        "bandwidth": 12,
                    },
                ),
                (
                    "driscoll_kraay_24",
                    {
                        "cov_type": "kernel",
                        "kernel": "bartlett",
                        "bandwidth": 24,
                    },
                ),
            ]
            for covariance, keywords in covariance_specs:
                fit = model.fit(debiased=True, **keywords)
                alpha_monthly = float(fit.params["constant"])
                standard_error_monthly = float(
                    fit.std_errors["constant"]
                )
                t_statistic = alpha_monthly / standard_error_monthly
                rows.append(
                    {
                        "test": key[0],
                        "comparator": key[1],
                        "period": key[2],
                        "scope": scope,
                        "covariance": covariance,
                        "markets": sample["market"].nunique(),
                        "observations": len(sample),
                        "start": sample["month"].min(),
                        "end": sample["month"].max(),
                        "alpha_annualized_sharpe": (
                            alpha_monthly * np.sqrt(12)
                        ),
                        "standard_error_annualized": (
                            standard_error_monthly * np.sqrt(12)
                        ),
                        "t_statistic": t_statistic,
                        "p_one_sided_normal": float(norm.sf(t_statistic)),
                        "p_two_sided_normal": float(
                            2 * norm.sf(abs(t_statistic))
                        ),
                        "library_p_two_sided": float(
                            fit.pvalues["constant"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def panel_contrasts(monthly: pd.DataFrame) -> pd.DataFrame:
    """Estimate post-2000 minus pre-2000 differences in one joint model."""
    rows: list[dict[str, object]] = []
    keys = ["test", "comparator"]
    covariance_specs = [
        (
            "calendar_month_cluster",
            {"cov_type": "clustered", "cluster_time": True},
        ),
        (
            "driscoll_kraay_12",
            {
                "cov_type": "kernel",
                "kernel": "bartlett",
                "bandwidth": 12,
            },
        ),
        (
            "driscoll_kraay_24",
            {
                "cov_type": "kernel",
                "kernel": "bartlett",
                "bandwidth": 24,
            },
        ),
    ]
    for key, base in monthly.dropna(subset=["z"]).groupby(keys, sort=True):
        counts = (
            base.groupby(["market", "period"])["z"]
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["before_2000", "from_2000"], fill_value=0)
        )
        for scope, eligible in [
            (
                "all_available",
                counts[
                    counts["before_2000"].gt(0)
                    & counts["from_2000"].gt(0)
                ].index,
            ),
            (
                "minimum_60_months_each_era",
                counts[
                    counts["before_2000"].ge(60)
                    & counts["from_2000"].ge(60)
                ].index,
            ),
        ]:
            sample = base[base["market"].isin(eligible)].copy()
            sample["time"] = pd.PeriodIndex(
                sample["month"],
                freq="M",
            ).to_timestamp()
            sample["post_2000"] = sample["period"].eq("from_2000").astype(float)
            indexed = sample.set_index(["market", "time"]).sort_index()
            exog = pd.DataFrame(
                {
                    "constant": 1.0,
                    "post_2000": indexed["post_2000"],
                },
                index=indexed.index,
            )
            model = PanelOLS(indexed["z"].astype(float), exog)
            before_observations = int(
                sample["period"].eq("before_2000").sum()
            )
            after_observations = int(
                sample["period"].eq("from_2000").sum()
            )
            for covariance, keywords in covariance_specs:
                fit = model.fit(debiased=True, **keywords)
                pre_monthly = float(fit.params["constant"])
                contrast_monthly = float(fit.params["post_2000"])
                post_monthly = pre_monthly + contrast_monthly
                standard_error_monthly = float(
                    fit.std_errors["post_2000"]
                )
                t_statistic = contrast_monthly / standard_error_monthly
                rows.append(
                    {
                        "test": key[0],
                        "comparator": key[1],
                        "contrast": "from_2000_minus_before_2000",
                        "scope": scope,
                        "covariance": covariance,
                        "markets": sample["market"].nunique(),
                        "observations": len(sample),
                        "before_2000_observations": before_observations,
                        "from_2000_observations": after_observations,
                        "start": sample["month"].min(),
                        "end": sample["month"].max(),
                        "before_2000_alpha_annualized_sharpe": (
                            pre_monthly * np.sqrt(12)
                        ),
                        "from_2000_alpha_annualized_sharpe": (
                            post_monthly * np.sqrt(12)
                        ),
                        "difference_annualized_sharpe": (
                            contrast_monthly * np.sqrt(12)
                        ),
                        "standard_error_difference_annualized": (
                            standard_error_monthly * np.sqrt(12)
                        ),
                        "t_statistic": t_statistic,
                        "p_one_sided_positive_normal": float(
                            norm.sf(t_statistic)
                        ),
                        "p_two_sided_normal": float(
                            2 * norm.sf(abs(t_statistic))
                        ),
                        "library_p_two_sided": float(
                            fit.pvalues["post_2000"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def validate_contrasts(
    inference: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> None:
    """Reconcile every joint-model era mean with the separate estimates."""
    keys = ["test", "comparator", "scope", "covariance"]
    era = inference.pivot(
        index=keys,
        columns="period",
        values="alpha_annualized_sharpe",
    )
    for row in contrasts.itertuples(index=False):
        if row.scope != "all_available":
            continue
        key = (row.test, row.comparator, row.scope, row.covariance)
        expected_pre = float(era.loc[key, "before_2000"])
        expected_post = float(era.loc[key, "from_2000"])
        expected_difference = expected_post - expected_pre
        checks = [
            (
                row.before_2000_alpha_annualized_sharpe,
                expected_pre,
                "pre-2000 alpha",
            ),
            (
                row.from_2000_alpha_annualized_sharpe,
                expected_post,
                "post-2000 alpha",
            ),
            (
                row.difference_annualized_sharpe,
                expected_difference,
                "post-minus-pre difference",
            ),
        ]
        for observed, expected, label in checks:
            if abs(float(observed) - expected) > 1e-12:
                raise RuntimeError(
                    f"Contrast reconciliation failed for {key}: {label}"
                )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    countries, monthly = collect()
    summary = summarize_countries(countries)
    inference = panel_inference(monthly)
    contrasts = panel_contrasts(monthly)
    validate_contrasts(inference, contrasts)
    paths = [
        OUT / COUNTRY_NAME,
        OUT / SUMMARY_NAME,
        OUT / MONTHLY_NAME,
        OUT / INFERENCE_NAME,
        OUT / CONTRAST_NAME,
    ]
    countries.to_csv(paths[0], index=False)
    summary.to_csv(paths[1], index=False)
    monthly.to_csv(paths[2], index=False)
    inference.to_csv(paths[3], index=False)
    contrasts.to_csv(paths[4], index=False)
    manifest = pd.DataFrame(
        [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
    )
    manifest.to_csv(OUT / MANIFEST_NAME, index=False)

    print("FULL-HISTORY +0.147 RECONCILIATION: PASS")
    print("FULL-HISTORY PORTFOLIO RECONSTRUCTION: PASS")
    print(
        summary[summary["scope"].eq("all_available")][
            [
                "test",
                "comparator",
                "period",
                "markets",
                "positive_markets",
                "mean_delta_sharpe",
                "median_delta_sharpe",
                "sign_test_p_one_sided",
            ]
        ].to_string(index=False)
    )
    print(
        inference[
            inference["scope"].eq("all_available")
            & inference["covariance"].eq("driscoll_kraay_12")
        ][
            [
                "test",
                "comparator",
                "period",
                "markets",
                "observations",
                "alpha_annualized_sharpe",
                "t_statistic",
                "p_one_sided_normal",
                "p_two_sided_normal",
            ]
        ].to_string(index=False)
    )
    print(
        contrasts[
            contrasts["scope"].eq("all_available")
            & contrasts["covariance"].isin(
                ["driscoll_kraay_12", "driscoll_kraay_24"]
            )
            & contrasts["test"].eq("B")
            & contrasts["comparator"].eq("era_static_monthly")
        ][
            [
                "test",
                "comparator",
                "covariance",
                "markets",
                "observations",
                "difference_annualized_sharpe",
                "t_statistic",
                "p_one_sided_positive_normal",
                "p_two_sided_normal",
            ]
        ].to_string(index=False)
    )
    print(f"wrote {len(paths)} outputs and {MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
