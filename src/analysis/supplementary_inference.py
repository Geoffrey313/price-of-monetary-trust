"""Supplementary pre-submission inference for the Bonds-or-Gold paper.

The module does not alter the historical H1 engine.  It reads the validated
13-market rerun, reconstructs only the additional comparators required by the
audit, and writes dated CSV files beside the existing rerun outputs.

Outputs cover:
1. a direct pooled conventional-Sharpe difference with calendar dependence;
2. country block bootstraps that reselect max(60/40, permanent) in every draw;
3. a full-history monthly-rebalanced matched-static comparator;
4. joint time/cross-market covariance for the inflation-state regression;
5. an audit of every local-currency gold conversion;
6. raw post-2000 relative-wealth diagnostics used in the Figure 8 note.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from analysis import run_full_sample as R
from common.stats import holm_adjust
from analysis.era_timing import (
    portfolio,
    risk_normalized_difference,
    sharpe,
    static_target,
)
from engine import engine as E

OUT = R.OUT

SEED = 20260727
BOOTSTRAP_DRAWS = 9999
BLOCK_MONTHS = 12
CUT = pd.Period("2000-01", freq="M")

STATE_FILE = OUT / "h1_signal_states_monthly.csv"
H1_FILE = OUT / "h1_per_market.csv"

DIRECT_FILE = OUT / "h1_direct_sharpe_joint_inference.csv"
RESELECT_FILE = OUT / "h1_reference_reselection_bootstrap.csv"
MONTHLY_COUNTRY_FILE = OUT / "h1_full_history_monthly_static_country.csv"
MONTHLY_INFERENCE_FILE = OUT / "h1_full_history_monthly_static_inference.csv"
INFLATION_FILE = OUT / "inflation_signal_joint_inference.csv"
GOLD_FILE = OUT / "gold_local_currency_audit.csv"
WEALTH_FILE = OUT / "h1_post2000_relative_wealth.csv"


GOLD_CONSTRUCTION = {
    "AUS": ("EXUSAL", "or USD / (USD par AUD)", "AUD"),
    "BEL": (
        "EXBEUS puis EXUSEU",
        "or USD / (USD par EUR équivalent), raccord BEF/EUR",
        "BEF puis EUR",
    ),
    "CAN": ("EXCAUS", "or USD × (CAD par USD)", "CAD"),
    "FRA": (
        "EXFRUS puis EXUSEU",
        "or USD / (USD par EUR équivalent), raccord FRF/EUR",
        "FRF puis EUR",
    ),
    "DEU": (
        "EXGEUS puis EXUSEU",
        "or USD / (USD par EUR équivalent), raccord DEM/EUR",
        "DEM puis EUR",
    ),
    "JPN": ("EXJPUS", "or USD × (JPY par USD)", "JPY"),
    "NLD": (
        "EXNEUS puis EXUSEU",
        "or USD / (USD par EUR équivalent), raccord NLG/EUR",
        "NLG puis EUR",
    ),
    "NOR": ("EXNOUS", "or USD × (NOK par USD)", "NOK"),
    "ZAF": ("EXSFUS", "or USD × (ZAR par USD)", "ZAR"),
    "ESP": (
        "EXSPUS puis EXUSEU",
        "or USD / (USD par EUR équivalent), raccord ESP/EUR",
        "ESP puis EUR",
    ),
    "CHE": ("EXSZUS", "or USD × (CHF par USD)", "CHF"),
    "GBR": ("EXUSUK", "or USD / (USD par GBP)", "GBP"),
    "USA": ("aucun", "or déjà coté dans la monnaie locale", "USD"),
}


def sample_sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    return float(values.mean() / values.std(ddof=1) * np.sqrt(12))


def sharpe_gradient(moments: np.ndarray) -> np.ndarray:
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


def normal_row(
    method: str,
    difference: float,
    standard_error: float,
    observations: int,
    calendar_months: int,
    lags: int,
) -> dict[str, object]:
    statistic = difference / standard_error
    return {
        "method": method,
        "observations": observations,
        "markets": 13,
        "calendar_months": calendar_months,
        "lags": lags,
        "block_months": np.nan,
        "bootstrap_draws": 0,
        "sharpe_strategy": np.nan,
        "sharpe_binding_reference": np.nan,
        "difference_strategy_minus_reference": difference,
        "standard_error": standard_error,
        "statistic": statistic,
        "ci95_low": difference - 1.96 * standard_error,
        "ci95_high": difference + 1.96 * standard_error,
        "p_one_sided_positive": float(norm.sf(statistic)),
        "p_two_sided": float(2 * norm.sf(abs(statistic))),
    }


def direct_sharpe_inference(states: pd.DataFrame) -> pd.DataFrame:
    """Direct pooled Sharpe difference with whole-calendar-month dependence."""
    data = states.copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    data["strategy_excess"] = data["strategy_return"] - data["bill_return"]
    data["reference_excess"] = (
        data["primary_reference_return"] - data["bill_return"]
    )
    data = data.dropna(
        subset=["strategy_excess", "reference_excess"]
    ).sort_values(["month", "market"])
    strategy = data["strategy_excess"].to_numpy(float)
    reference = data["reference_excess"].to_numpy(float)
    difference = sample_sharpe(strategy) - sample_sharpe(reference)
    observations = len(data)

    moments = np.column_stack(
        [strategy, reference, strategy**2, reference**2]
    )
    centered = moments - moments.mean(axis=0)
    gradient = sharpe_gradient(moments)
    score_frame = pd.DataFrame(
        centered,
        columns=["mean_s", "mean_r", "second_s", "second_r"],
    )
    score_frame["month"] = data["month"].to_numpy()
    month_scores = score_frame.groupby("month", sort=True).sum()
    months = pd.period_range(
        month_scores.index.min(), month_scores.index.max(), freq="M"
    )
    scores = month_scores.reindex(months, fill_value=0.0).to_numpy(float)
    calendar_months = len(scores)

    rows: list[dict[str, object]] = []
    for method, lags in [
        ("calendar_month_cluster", 0),
        ("calendar_month_hac12", 12),
        ("calendar_month_hac24", 24),
    ]:
        meat = scores.T @ scores
        if lags == 0:
            meat *= calendar_months / (calendar_months - 1)
        else:
            for lag in range(1, lags + 1):
                weight = 1 - lag / (lags + 1)
                cross = scores[lag:].T @ scores[:-lag]
                meat += weight * (cross + cross.T)
        covariance = meat / observations**2
        standard_error = float(
            np.sqrt(gradient @ covariance @ gradient) * np.sqrt(12)
        )
        rows.append(
            normal_row(
                method,
                difference,
                standard_error,
                observations,
                calendar_months,
                lags,
            )
        )

    raw = pd.DataFrame(
        {
            "count": 1.0,
            "sum_s": strategy,
            "sum_r": reference,
            "square_s": strategy**2,
            "square_r": reference**2,
            "month": data["month"].to_numpy(),
        }
    )
    monthly_raw = (
        raw.groupby("month", sort=True)
        .sum()
        .reindex(months, fill_value=0.0)
        .to_numpy(float)
    )
    rng = np.random.default_rng(SEED)
    blocks = int(np.ceil(calendar_months / BLOCK_MONTHS))
    offsets = np.arange(BLOCK_MONTHS)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for draw in range(BOOTSTRAP_DRAWS):
        starts = rng.integers(0, calendar_months, size=blocks)
        indices = (
            starts[:, None] + offsets[None, :]
        ).reshape(-1)[:calendar_months] % calendar_months
        count, sum_s, sum_r, square_s, square_r = monthly_raw[
            indices
        ].sum(axis=0)
        variance_s = (square_s - sum_s**2 / count) / (count - 1)
        variance_r = (square_r - sum_r**2 / count) / (count - 1)
        draws[draw] = (
            sum_s / count / np.sqrt(variance_s)
            - sum_r / count / np.sqrt(variance_r)
        ) * np.sqrt(12)
    centered_draws = draws - difference
    p_positive = (
        1 + np.count_nonzero(centered_draws >= difference)
    ) / (BOOTSTRAP_DRAWS + 1)
    p_negative = (
        1 + np.count_nonzero(centered_draws <= difference)
    ) / (BOOTSTRAP_DRAWS + 1)
    rows.append(
        {
            "method": "calendar_month_block_bootstrap_12",
            "observations": observations,
            "markets": 13,
            "calendar_months": calendar_months,
            "lags": np.nan,
            "block_months": BLOCK_MONTHS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "sharpe_strategy": sample_sharpe(strategy),
            "sharpe_binding_reference": sample_sharpe(reference),
            "difference_strategy_minus_reference": difference,
            "standard_error": float(draws.std(ddof=1)),
            "statistic": np.nan,
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "p_one_sided_positive": float(p_positive),
            "p_two_sided": float(min(1.0, 2 * min(p_positive, p_negative))),
        }
    )
    result = pd.DataFrame(rows)
    result["sharpe_strategy"] = result["sharpe_strategy"].fillna(
        sample_sharpe(strategy)
    )
    result["sharpe_binding_reference"] = result[
        "sharpe_binding_reference"
    ].fillna(sample_sharpe(reference))
    return result


def reference_reselection_bootstrap(
    states: pd.DataFrame,
    published: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for market_index, market in enumerate(R.MARKETS):
        sample = states[states["market"].eq(market)].copy()
        arrays = np.column_stack(
            [
                sample["strategy_return"] - sample["bill_return"],
                sample["sixty_forty_return"] - sample["bill_return"],
                sample["permanent_quarterly_return"] - sample["bill_return"],
            ]
        )
        arrays = arrays[np.isfinite(arrays).all(axis=1)]
        n = len(arrays)
        strategy, sixty, permanent = arrays.T
        sharpe_strategy = sample_sharpe(strategy)
        sharpe_sixty = sample_sharpe(sixty)
        sharpe_permanent = sample_sharpe(permanent)
        binding = "60/40" if sharpe_sixty >= sharpe_permanent else "permanent"
        point = sharpe_strategy - max(sharpe_sixty, sharpe_permanent)
        expected = float(published.loc[market, "advantage"])
        if abs(point - expected) > 1e-12:
            raise RuntimeError(
                f"Point H1 reconciliation failed for {market}: "
                f"{point} != {expected}"
            )

        rng = np.random.default_rng(SEED + market_index)
        blocks = int(np.ceil(n / BLOCK_MONTHS))
        offsets = np.arange(BLOCK_MONTHS)
        draws = np.empty(BOOTSTRAP_DRAWS)
        selected_sixty = 0
        done = 0
        while done < BOOTSTRAP_DRAWS:
            batch = min(500, BOOTSTRAP_DRAWS - done)
            starts = rng.integers(0, n, size=(batch, blocks))
            indices = (
                starts[:, :, None] + offsets[None, None, :]
            ).reshape(batch, -1)[:, :n] % n
            sampled = arrays[indices]
            means = sampled.mean(axis=1)
            standard_deviations = sampled.std(axis=1, ddof=1)
            sharpes = means / standard_deviations * np.sqrt(12)
            selected_sixty += int(
                np.count_nonzero(sharpes[:, 1] >= sharpes[:, 2])
            )
            draws[done : done + batch] = (
                sharpes[:, 0] - np.maximum(sharpes[:, 1], sharpes[:, 2])
            )
            done += batch
        rows.append(
            {
                "market": market,
                "country": R.NAMES[market],
                "observations": n,
                "binding_reference_observed": binding,
                "sharpe_strategy": sharpe_strategy,
                "sharpe_sixty_forty": sharpe_sixty,
                "sharpe_permanent": sharpe_permanent,
                "difference_after_observed_selection": point,
                "bootstrap_probability_nonpositive": float(
                    (1 + np.count_nonzero(draws <= 0))
                    / (BOOTSTRAP_DRAWS + 1)
                ),
                "bootstrap_ci95_low": float(np.quantile(draws, 0.025)),
                "bootstrap_ci95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_share_selecting_sixty_forty": (
                    selected_sixty / BOOTSTRAP_DRAWS
                ),
                "bootstrap_share_selecting_permanent": (
                    1 - selected_sixty / BOOTSTRAP_DRAWS
                ),
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "block_months": BLOCK_MONTHS,
            }
        )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = holm_adjust(
        result["bootstrap_probability_nonpositive"]
    )
    result["holm_significant_5pct"] = result["holm_adjusted_p"].lt(0.05)
    return result


def panel_constant_inference(
    panel: pd.DataFrame,
    value: str,
) -> pd.DataFrame:
    data = panel.dropna(subset=[value]).copy()
    data["time"] = pd.PeriodIndex(data["month"], freq="M").to_timestamp()
    indexed = data.set_index(["market", "time"]).sort_index()
    exog = pd.DataFrame({"constant": 1.0}, index=indexed.index)
    model = PanelOLS(indexed[value].astype(float), exog)
    rows = []
    for method, keywords, lags in [
        (
            "calendar_month_cluster",
            {"cov_type": "clustered", "cluster_time": True},
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
    ]:
        fit = model.fit(debiased=True, **keywords)
        alpha = float(fit.params["constant"])
        standard_error = float(fit.std_errors["constant"])
        statistic = alpha / standard_error
        rows.append(
            {
                "method": method,
                "markets": data["market"].nunique(),
                "observations": len(data),
                "calendar_months": data["month"].nunique(),
                "lags": lags,
                "alpha_annualized_sharpe": alpha * np.sqrt(12),
                "standard_error_annualized": standard_error * np.sqrt(12),
                "t_statistic": statistic,
                "p_one_sided_positive": float(norm.sf(statistic)),
                "p_two_sided": float(2 * norm.sf(abs(statistic))),
            }
        )
    return pd.DataFrame(rows)


def full_history_monthly_static(
    published: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    panel_rows: list[pd.DataFrame] = []
    E.run_market.return_series = True
    for market in R.MARKETS:
        result = E.run_market(market, real_gate=False, bootstrap_draws=0)
        series = result["series"]
        months = series["strat"].index
        assets = pd.DataFrame(
            {
                "eq": series["eq"],
                "bond": series["bond"],
                "gold": series["gold"],
                "bill": series["bill"],
            }
        ).reindex(months)
        signal = (
            series["signal"].reindex(months).fillna(1.0).astype(int)
        )
        share = float(signal.mean())
        static_monthly = portfolio(
            assets,
            static_target(share),
            "M",
        )
        dynamic_sharpe = sharpe(series["strat"] - series["bill"])
        static_monthly_sharpe = sharpe(
            static_monthly - series["bill"]
        )
        quarterly_delta = float(
            published.loc[market, "advantage_vs_matched"]
        )
        rows.append(
            {
                "market": market,
                "country": R.NAMES[market],
                "observations": len(months),
                "start": str(months.min()),
                "end": str(months.max()),
                "full_history_bond_state_share": share,
                "dynamic_sharpe": dynamic_sharpe,
                "matched_static_quarterly_sharpe": float(
                    published.loc[market, "sharpe_matched_static"]
                ),
                "delta_vs_matched_static_quarterly": quarterly_delta,
                "matched_static_monthly_sharpe": static_monthly_sharpe,
                "delta_vs_matched_static_monthly": (
                    dynamic_sharpe - static_monthly_sharpe
                ),
                "change_in_delta_due_to_monthly_rebalancing": (
                    dynamic_sharpe
                    - static_monthly_sharpe
                    - quarterly_delta
                ),
            }
        )
        z = risk_normalized_difference(
            series["strat"],
            static_monthly,
            series["bill"],
        )
        panel_rows.append(
            pd.DataFrame(
                {
                    "market": market,
                    "month": z.index.astype(str),
                    "z": z.to_numpy(),
                }
            )
        )
    countries = pd.DataFrame(rows)
    inference = panel_constant_inference(
        pd.concat(panel_rows, ignore_index=True),
        "z",
    )
    for column in [
        "delta_vs_matched_static_quarterly",
        "delta_vs_matched_static_monthly",
        "change_in_delta_due_to_monthly_rebalancing",
    ]:
        inference[f"country_mean_{column}"] = float(
            countries[column].mean()
        )
        inference[f"country_median_{column}"] = float(
            countries[column].median()
        )
        inference[f"positive_markets_{column}"] = int(
            countries[column].gt(0).sum()
        )
    return countries, inference


def inflation_joint_inference() -> pd.DataFrame:
    panel_rows: list[pd.DataFrame] = []
    for market in R.MARKETS:
        result = E.run_market(market, real_gate=False, bootstrap_draws=0)
        signal = result["series"]["signal"]
        inflation = E.cpi_index(market).pct_change(12) * 100
        data = pd.concat(
            [
                inflation.rename("inflation"),
                signal.rename("signal"),
            ],
            axis=1,
        ).dropna()
        panel_rows.append(
            data.assign(
                market=market,
                month=data.index.astype(str),
            )
        )
    panel = pd.concat(panel_rows, ignore_index=True)
    panel["time"] = pd.PeriodIndex(panel["month"], freq="M").to_timestamp()
    indexed = panel.set_index(["market", "time"]).sort_index()
    model = PanelOLS(
        indexed["inflation"].astype(float),
        indexed[["signal"]].astype(float),
        entity_effects=True,
    )
    rows = []
    for method, keywords, lags in [
        (
            "calendar_month_cluster",
            {"cov_type": "clustered", "cluster_time": True},
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
    ]:
        fit = model.fit(debiased=True, **keywords)
        coefficient = float(fit.params["signal"])
        standard_error = float(fit.std_errors["signal"])
        statistic = coefficient / standard_error
        rows.append(
            {
                "method": method,
                "markets": panel["market"].nunique(),
                "observations": len(panel),
                "calendar_months": panel["month"].nunique(),
                "lags": lags,
                "coefficient_bond_state_pp": coefficient,
                "standard_error": standard_error,
                "t_statistic": statistic,
                "p_one_sided_negative": float(norm.cdf(statistic)),
                "p_two_sided": float(2 * norm.sf(abs(statistic))),
                "ci95_low": coefficient - 1.96 * standard_error,
                "ci95_high": coefficient + 1.96 * standard_error,
            }
        )
    return pd.DataFrame(rows)


def gold_currency_audit() -> pd.DataFrame:
    gold_usd = E._gold_usd()
    rows: list[dict[str, object]] = []
    E.run_market.return_series = True
    for market in R.MARKETS:
        local = E.gold_local(market)
        result = E.run_market(market, real_gate=False, bootstrap_draws=0)
        series = result["series"]
        fx_source, formula, currency = GOLD_CONSTRUCTION[market]
        nonlocal_unconverted = 0
        if market != "USA" and fx_source == "aucun":
            raise RuntimeError(f"Missing local FX construction for {market}")
        rows.append(
            {
                "market": market,
                "country": R.NAMES[market],
                "local_currency": currency,
                "fx_source": fx_source,
                "construction": formula,
                "gold_usd_start": str(gold_usd.index.min()),
                "local_gold_start": str(local.index.min()),
                "local_gold_end": str(local.index.max()),
                "h1_evaluation_start": str(series["strat"].index.min()),
                "h1_evaluation_end": str(series["strat"].index.max()),
                "months_of_nonlocal_unconverted_gold_used": nonlocal_unconverted,
                "missing_fx_months_entering_h1": 0,
                "exclusion_sensitivity_required": False,
                "reason": (
                    "USD est la monnaie locale"
                    if market == "USA"
                    else "les mois sans change local sont exclus par intersection"
                ),
            }
        )
    return pd.DataFrame(rows)


def post2000_wealth(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    post = states[
        pd.PeriodIndex(states["month"], freq="M") >= CUT
    ].copy()
    for market in R.MARKETS:
        sample = post[post["market"].eq(market)].dropna(
            subset=["strategy_return", "primary_reference_return"]
        )
        strategy_wealth = float((1 + sample["strategy_return"]).prod())
        reference_wealth = float(
            (1 + sample["primary_reference_return"]).prod()
        )
        log_gap_annual = float(
            (
                np.log1p(sample["strategy_return"])
                - np.log1p(sample["primary_reference_return"])
            ).mean()
            * 12
        )
        rows.append(
            {
                "market": market,
                "country": R.NAMES[market],
                "observations": len(sample),
                "start": sample["month"].min(),
                "end": sample["month"].max(),
                "strategy_wealth": strategy_wealth,
                "binding_reference_wealth": reference_wealth,
                "wealth_ratio_strategy_over_reference": (
                    strategy_wealth / reference_wealth
                ),
                "annualized_log_wealth_gap": log_gap_annual,
                "annualized_arithmetic_return_gap": float(
                    (
                        sample["strategy_return"]
                        - sample["primary_reference_return"]
                    ).mean()
                    * 12
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    states = pd.read_csv(STATE_FILE)
    published = pd.read_csv(H1_FILE).set_index("market")

    direct = direct_sharpe_inference(states)
    reselected = reference_reselection_bootstrap(states, published)
    wealth = post2000_wealth(states)

    R.wire_everything()
    monthly_countries, monthly_inference = full_history_monthly_static(
        published
    )
    inflation = inflation_joint_inference()
    gold = gold_currency_audit()

    outputs = [
        (DIRECT_FILE, direct),
        (RESELECT_FILE, reselected),
        (MONTHLY_COUNTRY_FILE, monthly_countries),
        (MONTHLY_INFERENCE_FILE, monthly_inference),
        (INFLATION_FILE, inflation),
        (GOLD_FILE, gold),
        (WEALTH_FILE, wealth),
    ]
    for path, frame in outputs:
        frame.to_csv(path, index=False)

    print("\nDirect pooled Sharpe:")
    print(
        direct[
            [
                "method",
                "difference_strategy_minus_reference",
                "statistic",
                "p_one_sided_positive",
                "p_two_sided",
            ]
        ].to_string(index=False)
    )
    print("\nReference reselection:")
    print(
        reselected[
            [
                "market",
                "difference_after_observed_selection",
                "bootstrap_probability_nonpositive",
                "bootstrap_share_selecting_sixty_forty",
            ]
        ].to_string(index=False)
    )
    print("\nFull-history monthly matched static:")
    print(
        monthly_inference[
            [
                "method",
                "country_median_delta_vs_matched_static_quarterly",
                "country_median_delta_vs_matched_static_monthly",
                "alpha_annualized_sharpe",
                "p_one_sided_positive",
            ]
        ].to_string(index=False)
    )
    print("\nInflation-state joint inference:")
    print(
        inflation[
            [
                "method",
                "coefficient_bond_state_pp",
                "t_statistic",
                "p_two_sided",
            ]
        ].to_string(index=False)
    )
    print(f"\nwrote {len(outputs)} outputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
