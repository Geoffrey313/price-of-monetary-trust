"""Estimate the descriptive H1 panel split before and after January 2000.

Inputs: the published H1 risk-adjusted monthly panel.
Outputs: market means and pooled inference CSV files in the complete-sample
result directory.
Purpose: quantify the paper's ex-post era split without assigning causality.

The split is descriptive and was chosen after inspection of the main results.
It does not identify the euro, inflation targeting, the Great Moderation, or
any other macroeconomic mechanism.
"""
from __future__ import annotations


import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from common.names import EURO, MARKETS, NON_EURO
from common.paths import FULL_SAMPLE_RESULTS, MACRO_RESULTS

OUT = FULL_SAMPLE_RESULTS
CONTROL_OUT = MACRO_RESULTS
PANEL = OUT / "h1_risk_adjusted_panel.csv"



def panel_index(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["time"] = pd.PeriodIndex(
        result["month"], freq="M"
    ).to_timestamp()
    return result.set_index(["market", "time"]).sort_index()


def pooled_rows(
    data: pd.DataFrame,
    group: str,
    period: str,
) -> list[dict[str, object]]:
    indexed = panel_index(data)
    exog = pd.DataFrame({"constant": 1.0}, index=indexed.index)
    model = PanelOLS(indexed["z"].astype(float), exog)
    covariance_specs = [
        (
            "calendar_month_cluster",
            {"cov_type": "clustered", "cluster_time": True},
        ),
        (
            "driscoll_kraay_12",
            {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 12},
        ),
        (
            "driscoll_kraay_24",
            {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 24},
        ),
    ]
    rows = []
    for covariance, keywords in covariance_specs:
        fit = model.fit(debiased=True, **keywords)
        alpha_monthly = float(fit.params["constant"])
        standard_error_monthly = float(fit.std_errors["constant"])
        t_statistic = alpha_monthly / standard_error_monthly
        rows.append(
            {
                "group": group,
                "period": period,
                "covariance": covariance,
                "observations": len(data),
                "markets": data["market"].nunique(),
                "start": data["month"].min(),
                "end": data["month"].max(),
                "alpha_monthly": alpha_monthly,
                "alpha_annualized_sharpe": alpha_monthly * np.sqrt(12),
                "standard_error_monthly": standard_error_monthly,
                "standard_error_annualized": (
                    standard_error_monthly * np.sqrt(12)
                ),
                "t_statistic": t_statistic,
                "p_one_sided_normal": float(norm.sf(t_statistic)),
                "p_two_sided_normal": float(
                    2 * norm.sf(abs(t_statistic))
                ),
                "library_p_two_sided": float(fit.pvalues["constant"]),
            }
        )
    return rows


def market_rows(data: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for period, sample in [
        ("before_2000", data[data["month"] < "2000-01"]),
        ("from_2000", data[data["month"] >= "2000-01"]),
    ]:
        for market, market_data in sample.groupby("market", sort=True):
            fit = sm.OLS(
                market_data["z"].to_numpy(float),
                np.ones((len(market_data), 1)),
            ).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": 12, "use_correction": True},
            )
            t_statistic = float(fit.tvalues[0])
            rows.append(
                {
                    "market": market,
                    "euro_group": market in EURO,
                    "period": period,
                    "observations": len(market_data),
                    "start": market_data["month"].min(),
                    "end": market_data["month"].max(),
                    "alpha_annualized_sharpe": (
                        float(fit.params[0]) * np.sqrt(12)
                    ),
                    "hac12_standard_error_annualized": (
                        float(fit.bse[0]) * np.sqrt(12)
                    ),
                    "hac12_t_statistic": t_statistic,
                    "hac12_p_one_sided_normal": float(
                        norm.sf(t_statistic)
                    ),
                    "positive_alpha": float(fit.params[0]) > 0,
                }
            )
    return rows


def reconcile(result: pd.DataFrame) -> None:
    full = pd.read_csv(
        OUT / "h1_joint_dependence_inference.csv"
    ).set_index("method")
    calculated = result[
        result["group"].eq("all_13")
        & result["period"].eq("full_history")
    ].set_index("covariance")
    for method in (
        "calendar_month_cluster",
        "driscoll_kraay_12",
        "driscoll_kraay_24",
    ):
        for column in (
            "alpha_annualized_sharpe",
            "standard_error_annualized",
            "t_statistic",
            "p_one_sided_normal",
        ):
            error = abs(
                float(calculated.loc[method, column])
                - float(full.loc[method, column])
            )
            if error > 1e-12:
                raise RuntimeError(
                    f"Full-history reconciliation failed: "
                    f"{method}/{column}/{error}"
                )

    controls = pd.read_csv(
        CONTROL_OUT / "h1_alpha_controls_comparison_2026-07-27.csv"
    )
    controls = controls[
        controls["column"].astype(str).eq("2")
        & controls["covariance"].eq("driscoll_kraay_12")
    ].iloc[0]
    calculated = result[
        result["group"].eq("all_13")
        & result["period"].eq("from_2000_to_2025")
        & result["covariance"].eq("driscoll_kraay_12")
    ].iloc[0]
    for column in (
        "observations",
        "alpha_annualized_sharpe",
        "p_one_sided_normal",
    ):
        error = abs(float(calculated[column]) - float(controls[column]))
        if error > 1e-12:
            raise RuntimeError(
                f"2000-2025 reconciliation failed: {column}/{error}"
            )


def main() -> int:
    data = pd.read_csv(PANEL).dropna(subset=["z"]).copy()
    if set(data["market"]) != set(MARKETS):
        raise RuntimeError("The H1 panel does not contain exactly 13 markets")
    if data.duplicated(["market", "month"]).any():
        raise RuntimeError("Duplicate market-month rows in the H1 panel")

    periods = [
        ("full_history", data),
        ("before_2000", data[data["month"] < "2000-01"]),
        ("from_2000", data[data["month"] >= "2000-01"]),
        (
            "from_2000_to_2025",
            data[data["month"].between("2000-01", "2025-12")],
        ),
    ]
    groups = [
        ("all_13", MARKETS),
        ("euro_5", EURO),
        ("non_euro_8", NON_EURO),
    ]
    rows: list[dict[str, object]] = []
    for period, period_data in periods:
        for group, markets in groups:
            sample = period_data[
                period_data["market"].isin(markets)
            ].copy()
            rows.extend(pooled_rows(sample, group, period))
    result = pd.DataFrame(rows)
    reconcile(result)
    result.to_csv(
        OUT / "post_2000_h1_inference_2026-07-27.csv",
        index=False,
    )

    markets = pd.DataFrame(market_rows(data))
    markets.to_csv(
        OUT / "post_2000_h1_market_means_2026-07-27.csv",
        index=False,
    )
    print(
        result[
            result["period"].isin(["before_2000", "from_2000"])
            & result["covariance"].eq("driscoll_kraay_12")
        ].to_string(index=False)
    )
    print(
        markets.groupby("period")["positive_alpha"]
        .agg(["sum", "count"])
        .to_string()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
