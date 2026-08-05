"""Focused inference corrections used by the French full-sample paper.

Outputs
-------
results/full-sample/h1_holm_13_markets.csv
results/full-sample/h1_joint_dependence_inference.csv
results/full-sample/h3_reverse_inference.csv
results/full-sample/rdd_1999_full_rank.csv

The pooled calendar-month H3 inference (``h3_pooled_calendar_inference.csv``) is
produced by ``analysis.run_full_sample`` (the ``run_h3`` stage) and is not
rewritten here.

The script does not redefine the sample or portfolio rules.  It adds:
1. joint-dependence inference for the pooled H1 constant, including two-way
   clustering and Driscoll-Kraay covariance estimates;
2. Holm-adjusted p-values for the thirteen descriptive market-level H1 tests;
3. two-sided and reverse-direction H3 inference, including a pooled covariance
   that allows contemporaneous cross-market dependence and twelve months of
   serial dependence in calendar time;
4. a full-rank local-linear 1999 specification with complete euro interactions and
   only markets with at least 36 risk-normalized months on both sides of the cutoff.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from analysis import run_full_sample as rerun
from analysis.h3_variant import run_variant, series_for
from common.stats import holm_adjust

EURO = rerun.EURO
MARKETS = rerun.MARKETS
NAMES = rerun.NAMES
OUT = rerun.OUT


def write_h1_holm() -> pd.DataFrame:
    h1 = pd.read_csv(OUT / "h1_per_market.csv")
    result = h1[
        ["market", "country", "advantage", "p_block_bootstrap"]
    ].copy()
    result["p_holm_13"] = holm_adjust(result["p_block_bootstrap"])
    result["nominal_5pct"] = result["p_block_bootstrap"] < 0.05
    result["holm_5pct"] = result["p_holm_13"] < 0.05
    result = result.sort_values("p_block_bootstrap").reset_index(drop=True)
    result.to_csv(OUT / "h1_holm_13_markets.csv", index=False)
    return result


def write_h1_joint_dependence_inference() -> pd.DataFrame:
    """Inference for the pooled H1 constant under both panel dimensions."""
    data = pd.read_csv(OUT / "h1_risk_adjusted_panel.csv")
    if set(data["market"]) != set(MARKETS):
        raise RuntimeError("H1 joint-dependence panel does not contain 13 markets")
    if data.duplicated(["market", "month"]).any():
        raise RuntimeError("Duplicate H1 market-month observations")
    data["time"] = pd.PeriodIndex(data["month"], freq="M").to_timestamp()
    indexed = data.set_index(["market", "time"]).sort_index()
    indexed["constant"] = 1.0
    model = PanelOLS(indexed["z"], indexed[["constant"]])
    specifications = [
        (
            "calendar_month_cluster",
            {"cov_type": "clustered", "cluster_time": True},
            0,
        ),
        (
            "country_cluster",
            {"cov_type": "clustered", "cluster_entity": True},
            0,
        ),
        (
            "country_calendar_two_way",
            {
                "cov_type": "clustered",
                "cluster_entity": True,
                "cluster_time": True,
            },
            0,
        ),
        (
            "driscoll_kraay_12",
            {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 12},
            12,
        ),
        (
            "driscoll_kraay_24",
            {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 24},
            24,
        ),
        (
            "driscoll_kraay_36",
            {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 36},
            36,
        ),
    ]
    rows = []
    for method, keywords, lags in specifications:
        fit = model.fit(debiased=True, **keywords)
        coefficient = float(fit.params["constant"])
        standard_error = float(fit.std_errors["constant"])
        t_statistic = coefficient / standard_error
        rows.append(
            {
                "method": method,
                "observations": len(data),
                "markets": data["market"].nunique(),
                "calendar_months": data["month"].nunique(),
                "lags": lags,
                "alpha_monthly": coefficient,
                "alpha_annualized_sharpe": coefficient * np.sqrt(12),
                "standard_error_monthly": standard_error,
                "standard_error_annualized": standard_error * np.sqrt(12),
                "t_statistic": t_statistic,
                "p_one_sided_normal": float(norm.sf(t_statistic)),
                "p_two_sided_normal": float(
                    2 * norm.sf(abs(t_statistic))
                ),
                "library_p_two_sided": float(fit.pvalues["constant"]),
            }
        )
    result = pd.DataFrame(rows)

    # Independent score-sum implementation of the calendar cluster and
    # Bartlett HAC. It preserves every market observed in a calendar month.
    residual = data["z"].to_numpy(float) - data["z"].mean()
    scores = (
        pd.DataFrame({"month": data["month"], "score": residual})
        .groupby("month", sort=True)["score"]
        .sum()
        .to_numpy(float)
    )
    observations = len(data)
    for lags, method in [
        (0, "calendar_month_cluster"),
        (12, "driscoll_kraay_12"),
        (24, "driscoll_kraay_24"),
        (36, "driscoll_kraay_36"),
    ]:
        meat = float(scores @ scores)
        for lag in range(1, lags + 1):
            weight = 1 - lag / (lags + 1)
            meat += 2 * weight * float(scores[lag:] @ scores[:-lag])
        manual_se = np.sqrt(meat / observations**2)
        position = result["method"].eq(method)
        result.loc[position, "score_sum_standard_error_monthly"] = manual_se
        result.loc[position, "absolute_se_crosscheck_error"] = abs(
            float(result.loc[position, "standard_error_monthly"].iloc[0])
            - manual_se
        )
    maximum_error = result["absolute_se_crosscheck_error"].dropna().max()
    if maximum_error > 2e-6:
        raise RuntimeError(
            f"H1 Driscoll-Kraay cross-check failed: {maximum_error}"
        )
    result.to_csv(OUT / "h1_joint_dependence_inference.csv", index=False)
    return result


def sharpe_difference_inference(
    amplitude: pd.Series,
    binary: pd.Series,
    bill: pd.Series,
    lags: int = 12,
) -> dict[str, float | int]:
    """HAC delta-method inference for annualized Sharpe(amplitude)-Sharpe(binary)."""
    e1 = (amplitude - bill).dropna()
    e2 = (binary - bill).reindex(e1.index)
    n = len(e1)
    difference = (
        e1.mean() / e1.std() - e2.mean() / e2.std()
    ) * np.sqrt(12)

    moments = np.column_stack([e1, e2, e1**2, e2**2])
    centered = moments - moments.mean(axis=0)
    long_run = centered.T @ centered / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        covariance = centered[lag:].T @ centered[:-lag] / n
        long_run += weight * (covariance + covariance.T)

    mean_1, mean_2, second_1, second_2 = moments.mean(axis=0)
    variance_1 = second_1 - mean_1**2
    variance_2 = second_2 - mean_2**2
    gradient = np.array(
        [
            second_1 / variance_1**1.5,
            -second_2 / variance_2**1.5,
            -0.5 * mean_1 / variance_1**1.5,
            0.5 * mean_2 / variance_2**1.5,
        ]
    )
    standard_error = float(
        np.sqrt(gradient @ long_run @ gradient / n) * np.sqrt(12)
    )
    z_stat = float(difference / standard_error)
    p_amplitude_greater = float(1 - norm.cdf(z_stat))
    p_binary_greater = float(norm.cdf(z_stat))
    p_two_sided = float(2 * min(p_amplitude_greater, p_binary_greater))
    return {
        "observations": n,
        "difference_amplitude_minus_binary": float(difference),
        "standard_error": standard_error,
        "z_statistic": z_stat,
        "ci95_low": float(difference - 1.96 * standard_error),
        "ci95_high": float(difference + 1.96 * standard_error),
        "p_amplitude_greater": p_amplitude_greater,
        "p_binary_greater": p_binary_greater,
        "p_two_sided": p_two_sided,
        "inference_method": f"market_hac{lags}",
    }


def _sharpe_gradient(moments: np.ndarray) -> np.ndarray:
    """Gradient of annualized Sharpe(amplitude)-Sharpe(binary)."""
    mean_1, mean_2, second_1, second_2 = moments.mean(axis=0)
    variance_1 = second_1 - mean_1**2
    variance_2 = second_2 - mean_2**2
    return np.array(
        [
            second_1 / variance_1**1.5,
            -second_2 / variance_2**1.5,
            -0.5 * mean_1 / variance_1**1.5,
            0.5 * mean_2 / variance_2**1.5,
        ]
    )


def _sharpe_difference(amplitude: np.ndarray, binary: np.ndarray) -> float:
    """Annualized sample Sharpe difference, using the paper's ddof=1 convention."""
    return float(
        (
            amplitude.mean() / amplitude.std(ddof=1)
            - binary.mean() / binary.std(ddof=1)
        )
        * np.sqrt(12)
    )


def _normal_inference(
    difference: float,
    standard_error: float,
) -> dict[str, float]:
    z_stat = float(difference / standard_error)
    p_amplitude_greater = float(1 - norm.cdf(z_stat))
    p_binary_greater = float(norm.cdf(z_stat))
    return {
        "standard_error": standard_error,
        "z_statistic": z_stat,
        "ci95_low": float(difference - 1.96 * standard_error),
        "ci95_high": float(difference + 1.96 * standard_error),
        "p_amplitude_greater": p_amplitude_greater,
        "p_binary_greater": p_binary_greater,
        "p_two_sided": float(
            2 * min(p_amplitude_greater, p_binary_greater)
        ),
    }


def pooled_calendar_inference(
    panel: pd.DataFrame,
    lags: int = 12,
    bootstrap_draws: int = 4999,
    block_length: int = 12,
    seed: int = 20260726,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Pooled H3 inference preserving calendar-time cross-market dependence.

    The point estimate gives each available market-month one observation, as in
    the paper's pooled estimate.  Covariance estimation first sums centered
    moment conditions within calendar month.  The primary estimator then
    applies a Bartlett HAC with twelve monthly lags to those sums.  A circular
    moving-block bootstrap resamples whole calendar months, keeping every
    market observed in a selected month together.
    """
    required = {"market", "month", "amplitude", "binary"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"pooled H3 panel missing columns: {sorted(missing)}")

    data = panel.dropna(subset=["amplitude", "binary"]).copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data = data.sort_values(["month", "market"]).reset_index(drop=True)
    amplitude = data["amplitude"].to_numpy(float)
    binary = data["binary"].to_numpy(float)
    observations = len(data)
    difference = _sharpe_difference(amplitude, binary)

    moments = np.column_stack(
        [amplitude, binary, amplitude**2, binary**2]
    )
    centered = moments - moments.mean(axis=0)
    gradient = _sharpe_gradient(moments)
    centered_frame = pd.DataFrame(
        centered, columns=["mean_a", "mean_b", "second_a", "second_b"]
    )
    centered_frame["month"] = data["month"].to_numpy()
    month_sums = centered_frame.groupby("month", sort=True).sum()
    full_months = pd.period_range(
        month_sums.index.min(), month_sums.index.max(), freq="M"
    )
    month_sums = month_sums.reindex(full_months, fill_value=0.0)
    cluster_scores = month_sums.to_numpy(float)
    calendar_months = len(cluster_scores)

    cluster_meat = cluster_scores.T @ cluster_scores
    cluster_covariance = cluster_meat / observations**2
    cluster_covariance *= calendar_months / (calendar_months - 1)
    cluster_se = float(
        np.sqrt(gradient @ cluster_covariance @ gradient) * np.sqrt(12)
    )
    cluster_result = {
        "method": "calendar_month_cluster",
        "observations": observations,
        "calendar_months": calendar_months,
        "markets": int(data["market"].nunique()),
        "lags": 0,
        "block_length": np.nan,
        "bootstrap_draws": 0,
        "difference_amplitude_minus_binary": difference,
        **_normal_inference(difference, cluster_se),
    }

    hac_meat = cluster_meat.copy()
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        lagged = cluster_scores[lag:].T @ cluster_scores[:-lag]
        hac_meat += weight * (lagged + lagged.T)
    hac_covariance = hac_meat / observations**2
    hac_se = float(
        np.sqrt(gradient @ hac_covariance @ gradient) * np.sqrt(12)
    )
    hac_result = {
        "method": f"calendar_month_hac{lags}",
        "observations": observations,
        "calendar_months": calendar_months,
        "markets": int(data["market"].nunique()),
        "lags": lags,
        "block_length": np.nan,
        "bootstrap_draws": 0,
        "difference_amplitude_minus_binary": difference,
        **_normal_inference(difference, hac_se),
    }

    raw_frame = pd.DataFrame(
        {
            "count": np.ones(observations),
            "sum_a": amplitude,
            "sum_b": binary,
            "square_a": amplitude**2,
            "square_b": binary**2,
            "month": data["month"].to_numpy(),
        }
    )
    monthly_raw = raw_frame.groupby("month", sort=True).sum()
    monthly_raw = monthly_raw.reindex(full_months, fill_value=0.0)
    monthly_array = monthly_raw.to_numpy(float)
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(calendar_months / block_length))
    bootstrap_differences = np.empty(bootstrap_draws)
    offsets = np.arange(block_length)
    for draw in range(bootstrap_draws):
        starts = rng.integers(0, calendar_months, size=blocks_needed)
        sampled_months = (
            starts[:, None] + offsets[None, :]
        ).reshape(-1)[:calendar_months] % calendar_months
        totals = monthly_array[sampled_months].sum(axis=0)
        count, sum_a, sum_b, square_a, square_b = totals
        variance_a = (square_a - sum_a**2 / count) / (count - 1)
        variance_b = (square_b - sum_b**2 / count) / (count - 1)
        bootstrap_differences[draw] = (
            sum_a / count / np.sqrt(variance_a)
            - sum_b / count / np.sqrt(variance_b)
        ) * np.sqrt(12)

    bootstrap_se = float(bootstrap_differences.std(ddof=1))
    centered_bootstrap = bootstrap_differences - difference
    p_binary_greater = float(
        (
            1
            + np.count_nonzero(centered_bootstrap <= difference)
        )
        / (bootstrap_draws + 1)
    )
    p_amplitude_greater = float(
        (
            1
            + np.count_nonzero(centered_bootstrap >= difference)
        )
        / (bootstrap_draws + 1)
    )
    p_two_sided = float(
        min(1.0, 2 * min(p_binary_greater, p_amplitude_greater))
    )
    bootstrap_result = {
        "method": f"calendar_month_block_bootstrap_{block_length}",
        "observations": observations,
        "calendar_months": calendar_months,
        "markets": int(data["market"].nunique()),
        "lags": np.nan,
        "block_length": block_length,
        "bootstrap_draws": bootstrap_draws,
        "difference_amplitude_minus_binary": difference,
        "standard_error": bootstrap_se,
        "z_statistic": np.nan,
        "ci95_low": float(np.quantile(bootstrap_differences, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_differences, 0.975)),
        "p_amplitude_greater": p_amplitude_greater,
        "p_binary_greater": p_binary_greater,
        "p_two_sided": p_two_sided,
    }
    methods = pd.DataFrame(
        [cluster_result, hac_result, bootstrap_result]
    )
    return hac_result, methods


def write_h3_inference() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    pooled_parts: list[pd.DataFrame] = []
    for market in MARKETS:
        returns, binary_signal, amplitude_fraction = series_for(market)
        binary = run_variant(returns, binary_signal)
        amplitude = run_variant(returns, amplitude_fraction)
        inference = sharpe_difference_inference(
            amplitude, binary, returns["bill"]
        )
        rows.append(
            {
                "market": market,
                "country": NAMES[market],
                **inference,
            }
        )
        binary_excess = (binary - returns["bill"]).dropna()
        amplitude_excess = (amplitude - returns["bill"]).reindex(
            binary_excess.index
        )
        pooled_parts.append(
            pd.DataFrame(
                {
                    "market": market,
                    "month": binary_excess.index,
                    "amplitude": amplitude_excess.to_numpy(),
                    "binary": binary_excess.to_numpy(),
                }
            )
        )

    pooled_panel = pd.concat(pooled_parts, ignore_index=True)
    pooled, pooled_methods = pooled_calendar_inference(
        pooled_panel,
    )
    pooled_row = {
        "market": "POOLED",
        "country": "Groupé (mois calendaires)",
        "inference_method": pooled.pop("method"),
        "observations": pooled.pop("observations"),
        **pooled,
    }
    rows.append(pooled_row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "h3_reverse_inference.csv", index=False)
    # h3_pooled_calendar_inference.csv is owned by analysis.run_full_sample.
    return result, pooled_methods


def eligible_markets(
    data: pd.DataFrame, minimum_side: int = 36
) -> list[str]:
    counts = data.groupby("market")["running_month"].agg(
        pre=lambda x: int((x < 0).sum()),
        post=lambda x: int((x >= 0).sum()),
    )
    return counts[
        (counts["pre"] >= minimum_side) & (counts["post"] >= minimum_side)
    ].index.tolist()


def rdd_fit(
    panel: pd.DataFrame, bandwidth: int, euro_difference: bool
) -> dict[str, object]:
    data = panel.copy()
    periods = pd.PeriodIndex(data["month"], freq="M")
    data["running_month"] = (
        (periods.year - 1999) * 12 + periods.month - 1
    )
    data["post"] = (data["running_month"] >= 0).astype(float)
    data["euro"] = data["market"].isin(EURO).astype(float)
    markets = eligible_markets(data)
    data = data[
        data["market"].isin(markets)
        & (data["running_month"].abs() <= bandwidth)
    ].copy()

    design = pd.DataFrame(
        {
            "post": data["post"].to_numpy(),
            "running": data["running_month"].to_numpy(),
            "running_post": (
                data["running_month"] * data["post"]
            ).to_numpy(),
        }
    )
    dummies = pd.get_dummies(
        data["market"], prefix="market", drop_first=True, dtype=float
    ).reset_index(drop=True)
    design = pd.concat([design, dummies], axis=1)
    if euro_difference:
        design["post_euro"] = (
            data["post"] * data["euro"]
        ).to_numpy()
        design["running_euro"] = (
            data["running_month"] * data["euro"]
        ).to_numpy()
        design["running_post_euro"] = (
            data["running_month"] * data["post"] * data["euro"]
        ).to_numpy()
    design = sm.add_constant(design)

    matrix = design.to_numpy(dtype=float)
    fit = sm.OLS(data["z"].to_numpy(), matrix).fit(
        cov_type="cluster",
        cov_kwds={"groups": data["month"].to_numpy()},
    )
    names = list(design.columns)
    post_index = names.index("post")
    result: dict[str, object] = {
        "bandwidth_months": bandwidth,
        "euro_difference": euro_difference,
        "markets": len(markets),
        "market_codes": " ".join(markets),
        "observations": len(data),
        "design_columns": matrix.shape[1],
        "design_rank": int(np.linalg.matrix_rank(matrix)),
        "condition_number": float(np.linalg.cond(matrix)),
        "common_break_monthly": float(fit.params[post_index]),
        "common_break_t": float(fit.tvalues[post_index]),
        "common_break_p_two_sided": float(fit.pvalues[post_index]),
    }
    if euro_difference:
        euro_index = names.index("post_euro")
        result.update(
            {
                "additional_euro_break_monthly": float(
                    fit.params[euro_index]
                ),
                "additional_euro_break_t": float(
                    fit.tvalues[euro_index]
                ),
                "additional_euro_break_p_two_sided": float(
                    fit.pvalues[euro_index]
                ),
            }
        )
    return result


def write_rdd() -> pd.DataFrame:
    panel = pd.read_csv(OUT / "h1_risk_adjusted_panel.csv")
    rows = [
        rdd_fit(panel, bandwidth, euro_difference)
        for bandwidth in (48, 60, 84, 120)
        for euro_difference in (False, True)
    ]
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "rdd_1999_full_rank.csv", index=False)
    return result


def main() -> int:
    rerun.wire_everything()
    h1 = write_h1_holm()
    h1_joint = write_h1_joint_dependence_inference()
    h3, h3_pooled = write_h3_inference()
    rdd = write_rdd()
    print(
        "H1:",
        int(h1["nominal_5pct"].sum()),
        "nominal passes,",
        int(h1["holm_5pct"].sum()),
        "Holm passes",
    )
    print("H1 joint-dependence inference:")
    print(h1_joint.to_string(index=False))
    print("H3 pooled:")
    print(h3[h3["market"] == "POOLED"].to_string(index=False))
    print("H3 pooled inference checks:")
    print(h3_pooled.to_string(index=False))
    print("RDD at 60 months:")
    print(rdd[rdd["bandwidth_months"] == 60].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
