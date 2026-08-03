"""Targeted diagnostics for the complete-sample rerun.

This script audits two points that materially affect interpretation:

1. The manuscript defines the binding H1 benchmark by the higher Sharpe ratio,
   while three historical panel scripts selected it by higher mean return.
   Both selections are recomputed with and without the obsolete CPI end gate.
2. The H2 exposure slope is checked with HC3 errors, robust regression and a
   leave-one-country-out jackknife, because the Swiss one-firm sleeve is an
   influential observation.
"""
from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

from analysis import run_full_sample as R
from engine import engine as E

OUT = R.OUT


def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def panel_summary(parts: list[pd.DataFrame], value: str) -> dict:
    panel = pd.concat(parts, ignore_index=True)
    mean, se, t = R.cluster_t(panel, value)
    return {
        "observations": len(panel),
        "month_clusters": panel["month"].nunique(),
        "alpha_monthly": mean,
        "alpha_annualized": mean * (12 if value == "d" else np.sqrt(12)),
        "se_monthly": se,
        "t": t,
        "p_two_sided": 2 * (1 - phi(abs(t))),
        "p_one_sided_positive": 1 - phi(t),
    }


def h1_benchmark_audit() -> None:
    R.wire_everything()
    E.run_market.return_series = True
    summaries = []
    market_rows = []
    for real_gate in (True, False):
        correct_d, correct_z = [], []
        legacy_d, legacy_z = [], []
        for market in R.MARKETS:
            result = E.run_market(
                market, real_gate=real_gate, bootstrap_draws=0
            )
            series = result["series"]
            bill = series["bill"]
            sharpe_6040 = R.sharpe(series["b6040"], bill)
            sharpe_pp = R.sharpe(series["pp"], bill)
            mean_6040 = float((series["b6040"] - bill).mean())
            mean_pp = float((series["pp"] - bill).mean())
            correct_name = "60/40" if sharpe_6040 >= sharpe_pp else "permanent"
            legacy_name = "60/40" if mean_6040 >= mean_pp else "permanent"
            correct = series["b6040"] if correct_name == "60/40" else series["pp"]
            legacy = series["b6040"] if legacy_name == "60/40" else series["pp"]
            ex_strategy = series["strat"] - bill
            for benchmark, d_parts, z_parts in (
                (correct, correct_d, correct_z),
                (legacy, legacy_d, legacy_z),
            ):
                ex_benchmark = benchmark - bill
                d = (ex_strategy - ex_benchmark).dropna()
                z = (
                    ex_strategy / ex_strategy.rolling(36).std().shift(1)
                    - ex_benchmark / ex_benchmark.rolling(36).std().shift(1)
                ).dropna()
                d_parts.append(
                    pd.DataFrame(
                        {
                            "d": d.values,
                            "market": market,
                            "month": d.index.astype(str),
                        }
                    )
                )
                z_parts.append(
                    pd.DataFrame(
                        {
                            "z": z.values,
                            "market": market,
                            "month": z.index.astype(str),
                        }
                    )
                )
            market_rows.append(
                {
                    "real_gate": real_gate,
                    "market": market,
                    "sharpe_6040": sharpe_6040,
                    "sharpe_permanent": sharpe_pp,
                    "mean_excess_6040_monthly": mean_6040,
                    "mean_excess_permanent_monthly": mean_pp,
                    "correct_benchmark_higher_sharpe": correct_name,
                    "legacy_benchmark_higher_mean_return": legacy_name,
                    "selection_differs": correct_name != legacy_name,
                }
            )
        for selection, d_parts, z_parts in (
            ("higher_sharpe_manuscript_definition", correct_d, correct_z),
            ("higher_mean_legacy_code", legacy_d, legacy_z),
        ):
            for outcome, parts in (("return_level", d_parts), ("risk_adjusted", z_parts)):
                row = panel_summary(parts, "d" if outcome == "return_level" else "z")
                row.update(
                    {
                        "benchmark_selection": selection,
                        "cpi_end_gate": real_gate,
                        "outcome": outcome,
                    }
                )
                summaries.append(row)
    pd.DataFrame(summaries).to_csv(
        OUT / "h1_benchmark_selection_audit.csv", index=False
    )
    pd.DataFrame(market_rows).to_csv(
        OUT / "h1_benchmark_selection_by_market.csv", index=False
    )


def h2_influence_audit() -> None:
    h2 = pd.read_csv(OUT / "h2_per_market.csv")
    coverage = pd.read_csv(OUT / "h2_energy_coverage.csv")[
        ["market", "median_effective_n", "median_firms"]
    ]
    h2 = h2.merge(coverage, on="market", how="left")
    rows = []
    rng = np.random.default_rng(R.SEED)
    omissions = ["NONE", *h2["market"].tolist()]
    for omitted in omissions:
        data = h2 if omitted == "NONE" else h2[h2["market"] != omitted]
        x = data["exposure_mean_rank"].to_numpy(float)
        y = data["augmentation_advantage"].to_numpy(float)
        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        hc3 = sm.OLS(y, X).fit(cov_type="HC3")
        rho = float(spearmanr(x, y).statistic)
        draws = 20000 if omitted == "NONE" else 5000
        more = 0
        for _ in range(draws):
            more += spearmanr(rng.permutation(x), y).statistic >= rho
        rows.append(
            {
                "omitted_market": omitted,
                "markets": len(data),
                "negative_augmentations": int((y < 0).sum()),
                "mean_augmentation": float(y.mean()),
                "median_augmentation": float(np.median(y)),
                "ols_intercept": float(ols.params[0]),
                "ols_slope": float(ols.params[1]),
                "ols_slope_se": float(ols.bse[1]),
                "ols_slope_t": float(ols.tvalues[1]),
                "ols_slope_p_one_sided": float(1 - phi(float(ols.tvalues[1]))),
                "hc3_slope_se": float(hc3.bse[1]),
                "hc3_slope_t": float(hc3.tvalues[1]),
                "hc3_slope_p_one_sided": float(1 - phi(float(hc3.tvalues[1]))),
                "spearman_rho": rho,
                "spearman_permutation_p_one_sided": (more + 1) / (draws + 1),
                "all_fitted_negative": bool(np.all(ols.predict(X) < 0)),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "h2_influence_audit.csv", index=False)

    X = sm.add_constant(h2["exposure_mean_rank"].to_numpy(float))
    y = h2["augmentation_advantage"].to_numpy(float)
    robust = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    pd.DataFrame(
        [
            {
                "method": "Huber_RLM",
                "intercept": float(robust.params[0]),
                "slope": float(robust.params[1]),
                "slope_se": float(robust.bse[1]),
                "slope_t": float(robust.tvalues[1]),
                "slope_p_one_sided_normal": float(1 - phi(float(robust.tvalues[1]))),
            }
        ]
    ).to_csv(OUT / "h2_robust_regression.csv", index=False)


def main() -> int:
    h1_benchmark_audit()
    h2_influence_audit()
    print("wrote targeted H1/H2 audit tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
