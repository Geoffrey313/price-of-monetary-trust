"""Deflated Sharpe ratio and Harvey-Liu-Zhu multiple-testing haircut.

Inputs: the committed full-sample outputs written by ``analysis.run_full_sample``
and ``analysis.supplementary_inference`` under
``results/full-sample`` (the pooled risk-adjusted panel, the
pooled Driscoll-Kraay inference, the monthly signal-state series, and the direct
pooled Sharpe inference).
Outputs: ``deflated_sharpe_multiple_testing.csv``, ``deflated_sharpe_inputs.csv``
and ``deflated_sharpe_haircut.csv`` (the Holm/BHY/Bonferroni haircut table) beside
those outputs, plus a printed report.
Purpose: subject the paper's two headline advantages to the Bailey and Lopez de
Prado (2014) Deflated Sharpe Ratio and to a Harvey, Liu and Zhu (2016)
multiple-testing haircut, so the reported significance is stated after the number
of trials rather than in isolation.

The primary estimand is the pooled risk-adjusted advantage
:math:`\\alpha_z = +0.086` annualized Sharpe-type units (Driscoll-Kraay, twelve
lags). The co-primary estimand is the direct pooled Sharpe difference
:math:`+0.122` (calendar-month HAC, twelve lags). Neither number is recomputed
here; the module reuses the exact series the paper builds and deflates the
statistics the paper already reports.

Determinism: the Deflated Sharpe Ratio and the haircut are closed form, so the
module draws no random numbers. ``SEED`` matches the full-sample pipeline and is
used only if a caller extends the module with a simulated trial distribution.

Run: ``PYTHONPATH=src python -m analysis.deflated_sharpe``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

from common.paths import FULL_SAMPLE_RESULTS
from common.stats import holm_adjust

OUT = FULL_SAMPLE_RESULTS
SEED = 20260726

EULER_MASCHERONI = 0.5772156649015329

# ---------------------------------------------------------------------------
# Trial counts.
#
# N_STRUCTURAL: the paper's argument is that every operating parameter is
# inherited from prior work, not tuned in sample -- the 84-month signal window
# (a seven-year cycle), the 60/40 and permanent-portfolio benchmarks, and
# monthly rebalancing. Under that reading the only genuine researcher forks are
# the three registered hypotheses (H1 monetary switch, H2 energy sleeve, H3
# functional form) crossed with the two confirmatory lenses on the headline
# advantage (the risk-adjusted alpha_z and the direct Sharpe difference), giving
# a small structural count of 3 x 2 = 6.
#
# N_REPORTED: the conservative count. It is the number of distinct confirmatory
# p-values the paper actually reports for H1 and H3: 13 per-market H1 tests, the
# 6 pooled-H1 covariance methods, 13 per-market H3 tests, the 3 pooled-H3
# methods, and the 4 direct-Sharpe methods. The module rebuilds this vector from
# the result files and counts it, so the number is not asserted but read.
N_STRUCTURAL = 6
# A round conservative external anchor (Harvey, Liu and Zhu 2016 curate ~316
# published factor tests); reported for context only.
N_LITERATURE = 316


def expected_max_standard_normal(trials: int) -> float:
    """Expected maximum of ``trials`` i.i.d. standard normals (Bailey-LdP).

    Uses the two-term extreme-value approximation from Bailey and Lopez de Prado
    (2014, eq. for :math:`E[\\max]`):
    :math:`(1-\\gamma)\\,Z^{-1}(1-1/N) + \\gamma\\,Z^{-1}(1-1/(N e))`, with
    :math:`\\gamma` the Euler-Mascheroni constant.
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if trials == 1:
        return 0.0
    gamma = EULER_MASCHERONI
    z1 = norm.ppf(1.0 - 1.0 / trials)
    z2 = norm.ppf(1.0 - 1.0 / (trials * math.e))
    return (1.0 - gamma) * z1 + gamma * z2


def deflated_sharpe_from_t(t_observed: float, trials: int) -> float:
    """Deflated Sharpe Ratio as a probability, anchored on an observed t.

    The Deflated Sharpe Ratio is
    :math:`\\Phi\\big((\\widehat{SR}-SR^{*})/\\sigma_{\\widehat{SR}}\\big)`. When
    the threshold is built from the same estimation variance as the estimate,
    :math:`SR^{*}/\\sigma_{\\widehat{SR}}` equals the expected maximum of ``N``
    standard normals, so the statistic collapses to
    :math:`\\Phi(t_{\\text{obs}} - E[\\max_N])`. Feeding the paper's
    dependence-robust t (Driscoll-Kraay or calendar-month HAC) makes the
    deflation inherit the non-normality and the cross-market dependence already
    priced into that standard error.
    """
    return float(norm.cdf(t_observed - expected_max_standard_normal(trials)))


def deflated_sharpe_closed_form(
    sharpe_per_period: float,
    skewness: float,
    kurtosis_nonexcess: float,
    observations: int,
    trials: int,
) -> dict[str, float]:
    """Canonical Bailey-Lopez de Prado (2014) closed-form Deflated Sharpe Ratio.

    ``sharpe_per_period`` is the non-annualized Sharpe of the return stream;
    ``skewness`` and ``kurtosis_nonexcess`` are its third and fourth standardized
    moments (a Gaussian has kurtosis 3); ``observations`` is the number of return
    periods. The sampling variance of the Sharpe estimate is
    :math:`(1-\\gamma_3 SR + \\tfrac{\\gamma_4-1}{4} SR^2)/(T-1)`, and the
    threshold :math:`SR^{*}` is the expected maximum of ``trials`` Sharpe
    estimates with that variance.
    """
    variance = (
        1.0
        - skewness * sharpe_per_period
        + (kurtosis_nonexcess - 1.0) / 4.0 * sharpe_per_period**2
    ) / (observations - 1)
    standard_error = math.sqrt(variance)
    threshold = standard_error * expected_max_standard_normal(trials)
    probabilistic_sharpe = float(norm.cdf(sharpe_per_period / standard_error))
    deflated = float(norm.cdf((sharpe_per_period - threshold) / standard_error))
    return {
        "sharpe_per_period": sharpe_per_period,
        "sharpe_annualized": sharpe_per_period * math.sqrt(12),
        "skewness": skewness,
        "kurtosis_nonexcess": kurtosis_nonexcess,
        "observations": observations,
        "sharpe_standard_error": standard_error,
        "probabilistic_sharpe_ratio": probabilistic_sharpe,
        "threshold_sharpe_star_per_period": threshold,
        "threshold_sharpe_star_annualized": threshold * math.sqrt(12),
        "expected_max_standard_normal": expected_max_standard_normal(trials),
        "trials": trials,
        "deflated_sharpe_ratio": deflated,
    }


def series_moments(values: np.ndarray) -> dict[str, float]:
    """Sharpe (per period and annualized), skewness, non-excess kurtosis, T."""
    values = np.asarray(values, dtype=float)
    per_period = float(values.mean() / values.std(ddof=1))
    return {
        "sharpe_per_period": per_period,
        "sharpe_annualized": per_period * math.sqrt(12),
        "skewness": float(skew(values, bias=False)),
        "kurtosis_nonexcess": float(kurtosis(values, fisher=False, bias=False)),
        "observations": int(len(values)),
    }


def calendar_month_mean(frame: pd.DataFrame, value: str) -> np.ndarray:
    """Equal-weighted cross-market advantage per calendar month.

    Averaging the per-market advantage within each calendar month yields one
    self-financing overlay return per month, the natural single stream whose
    Sharpe the closed-form Deflated Sharpe Ratio can act on without stacking the
    thirteen markets into one artificial series.
    """
    grouped = frame.groupby("month", sort=True)[value].mean()
    return grouped.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Harvey-Liu-Zhu multiple-testing haircut.


def benjamini_yekutieli_adjusted(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli (2001) FDR adjusted p-values under dependence.

    This is the BHY procedure Harvey, Liu and Zhu (2016) recommend because it is
    valid under the arbitrary cross-test dependence a specification search
    produces. The harmonic factor ``c(N) = sum 1/i`` is the dependence penalty.
    """
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)
    harmonic = float(np.sum(1.0 / np.arange(1, count + 1)))
    order = np.argsort(p_values)[::-1]
    ranks = np.arange(count, 0, -1)
    scaled = harmonic * count / ranks * p_values[order]
    running = np.minimum.accumulate(scaled)
    running = np.minimum(running, 1.0)
    adjusted = np.empty(count)
    adjusted[order] = running
    return adjusted


def confirmatory_p_family() -> pd.DataFrame:
    """Assemble the paper's confirmatory one-sided p-value family from disk.

    The family is the union of the thirteen per-market H1 block-bootstrap tests,
    the six pooled-H1 covariance methods, the thirteen per-market H3 tests, the
    three pooled-H3 methods, and the four direct-Sharpe methods. Every entry is
    a one-sided p-value for the registered direction, the object a haircut acts
    on.
    """
    rows: list[dict[str, object]] = []

    h1 = pd.read_csv(OUT / "h1_per_market.csv")
    for row in h1.itertuples():
        rows.append(
            {"family": "H1_per_market", "label": row.market, "p": float(row.p_block_bootstrap)}
        )

    h1_pooled = pd.read_csv(OUT / "h1_joint_dependence_inference.csv")
    for row in h1_pooled.itertuples():
        rows.append(
            {"family": "H1_pooled", "label": row.method, "p": float(row.p_one_sided_normal)}
        )

    h3 = pd.read_csv(OUT / "h3_per_market.csv")
    for row in h3.itertuples():
        rows.append(
            {"family": "H3_per_market", "label": row.market, "p": float(row.p_amplitude_greater)}
        )

    h3_pooled = pd.read_csv(OUT / "h3_pooled_calendar_inference.csv")
    for row in h3_pooled.itertuples():
        rows.append(
            {"family": "H3_pooled", "label": row.method, "p": float(row.p_amplitude_greater)}
        )

    direct = pd.read_csv(
        OUT / "h1_direct_sharpe_joint_inference.csv"
    )
    for row in direct.itertuples():
        rows.append(
            {
                "family": "direct_sharpe",
                "label": row.method,
                "p": float(row.p_one_sided_positive),
            }
        )

    return pd.DataFrame(rows)


def haircut_for_focal(
    focal_p: float,
    focal_label: str,
    family: pd.DataFrame,
    trials: int,
) -> dict[str, object]:
    """Bonferroni, Holm and BHY adjusted p-values for one focal test.

    The focal p-value is embedded in the reported family so Holm and BHY see the
    true rank of the headline result among all trials. Bonferroni uses the trial
    count directly. The Bonferroni-implied minimum one-sided t at the five-percent
    level is also reported.
    """
    augmented = pd.concat(
        [family, pd.DataFrame([{"family": "focal", "label": focal_label, "p": focal_p}])],
        ignore_index=True,
    )
    p_vector = augmented["p"].to_numpy(dtype=float)
    focal_index = len(augmented) - 1

    holm = holm_adjust(pd.Series(p_vector)).to_numpy()[focal_index]
    bhy = benjamini_yekutieli_adjusted(p_vector)[focal_index]
    bonferroni = min(1.0, trials * focal_p)
    required_t = float(norm.ppf(1.0 - 0.05 / trials))
    return {
        "focal_label": focal_label,
        "focal_one_sided_p": focal_p,
        "trials_bonferroni": trials,
        "family_size_for_holm_bhy": len(augmented),
        "bonferroni_adjusted_p": bonferroni,
        "bonferroni_required_one_sided_t_5pct": required_t,
        "holm_adjusted_p": float(holm),
        "bhy_adjusted_p": float(bhy),
        "survives_5pct_bonferroni": bool(bonferroni < 0.05),
        "survives_5pct_holm": bool(holm < 0.05),
        "survives_5pct_bhy": bool(bhy < 0.05),
    }


# ---------------------------------------------------------------------------
# Estimand assembly.


def primary_estimand() -> dict[str, object]:
    """Pooled risk-adjusted advantage alpha_z (+0.086), Driscoll-Kraay twelve lags."""
    panel = pd.read_csv(OUT / "h1_risk_adjusted_panel.csv")
    monthly = calendar_month_mean(panel, "z")
    inference = pd.read_csv(OUT / "h1_joint_dependence_inference.csv")
    driscoll = inference.loc[inference["method"] == "driscoll_kraay_12"].iloc[0]
    return {
        "name": "primary_risk_adjusted_alpha_z",
        "advantage_annualized": float(driscoll["alpha_annualized_sharpe"]),
        "t_observed": float(driscoll["t_statistic"]),
        "p_one_sided": float(driscoll["p_one_sided_normal"]),
        "inference": "Driscoll-Kraay, 12 lags",
        "pooled_observations": int(driscoll["observations"]),
        "calendar_months": int(driscoll["calendar_months"]),
        "monthly_series": monthly,
    }


def coprimary_estimand() -> dict[str, object]:
    """Direct pooled Sharpe difference (+0.122), calendar-month HAC twelve lags."""
    states = pd.read_csv(OUT / "h1_signal_states_monthly.csv")
    states["strategy_excess"] = states["strategy_return"] - states["bill_return"]
    states["reference_excess"] = (
        states["primary_reference_return"] - states["bill_return"]
    )
    states = states.dropna(subset=["strategy_excess", "reference_excess"])
    states = states.assign(
        advantage_excess=states["strategy_excess"] - states["reference_excess"]
    )
    monthly = calendar_month_mean(states, "advantage_excess")
    strategy_stream = states["strategy_excess"].to_numpy(dtype=float)

    direct = pd.read_csv(OUT / "h1_direct_sharpe_joint_inference.csv")
    hac = direct.loc[direct["method"] == "calendar_month_hac12"].iloc[0]
    return {
        "name": "coprimary_direct_sharpe_difference",
        "advantage_annualized": float(hac["difference_strategy_minus_reference"]),
        "t_observed": float(hac["statistic"]),
        "p_one_sided": float(hac["p_one_sided_positive"]),
        "inference": "calendar-month HAC, 12 lags",
        "pooled_observations": int(hac["observations"]),
        "calendar_months": int(hac["calendar_months"]),
        "sharpe_strategy_annualized": float(hac["sharpe_strategy"]),
        "sharpe_reference_annualized": float(hac["sharpe_binding_reference"]),
        "monthly_series": monthly,
        "strategy_excess_pooled": strategy_stream,
    }


def evaluate(estimand: dict[str, object], trial_ladder: list[int]) -> pd.DataFrame:
    """Deflated Sharpe Ratio over a ladder of trial counts for one estimand."""
    moments = series_moments(estimand["monthly_series"])
    rows: list[dict[str, object]] = []
    for trials in trial_ladder:
        e_max = expected_max_standard_normal(trials)
        # Headline: deflate the paper's dependence-robust t.
        dsr_t = deflated_sharpe_from_t(estimand["t_observed"], trials)
        # Cross-check: closed form on the equal-weighted advantage stream.
        closed = deflated_sharpe_closed_form(
            moments["sharpe_per_period"],
            moments["skewness"],
            moments["kurtosis_nonexcess"],
            moments["observations"],
            trials,
        )
        rows.append(
            {
                "estimand": estimand["name"],
                "advantage_annualized": estimand["advantage_annualized"],
                "t_observed_paper": estimand["t_observed"],
                "trials_N": trials,
                "expected_max_standard_normal": e_max,
                "dsr_t_anchored": dsr_t,
                "implied_one_sided_p_after_N": 1.0 - dsr_t,
                "advantage_stream_sharpe_annualized": moments["sharpe_annualized"],
                "advantage_stream_skewness": moments["skewness"],
                "advantage_stream_kurtosis_nonexcess": moments["kurtosis_nonexcess"],
                "advantage_stream_T": moments["observations"],
                "dsr_closed_form_advantage_stream": closed["deflated_sharpe_ratio"],
                "psr_vs_zero": closed["probabilistic_sharpe_ratio"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    primary = primary_estimand()
    coprimary = coprimary_estimand()
    family = confirmatory_p_family()
    n_reported = len(family)

    trial_ladder = sorted({N_STRUCTURAL, 13, n_reported, 100, N_LITERATURE})

    dsr_primary = evaluate(primary, trial_ladder)
    dsr_coprimary = evaluate(coprimary, trial_ladder)
    dsr_table = pd.concat([dsr_primary, dsr_coprimary], ignore_index=True)

    haircut_rows = [
        haircut_for_focal(
            primary["p_one_sided"], "primary_alpha_z_DK12", family, N_STRUCTURAL
        ),
        haircut_for_focal(
            primary["p_one_sided"], "primary_alpha_z_DK12", family, n_reported
        ),
        haircut_for_focal(
            coprimary["p_one_sided"], "coprimary_direct_sharpe_HAC12", family, N_STRUCTURAL
        ),
        haircut_for_focal(
            coprimary["p_one_sided"], "coprimary_direct_sharpe_HAC12", family, n_reported
        ),
    ]
    haircut_table = pd.DataFrame(haircut_rows)

    inputs_table = pd.DataFrame(
        [
            {
                "estimand": estimand["name"],
                "advantage_annualized": estimand["advantage_annualized"],
                "inference": estimand["inference"],
                "t_observed": estimand["t_observed"],
                "p_one_sided": estimand["p_one_sided"],
                "pooled_observations": estimand["pooled_observations"],
                "calendar_months": estimand["calendar_months"],
                **{
                    f"advantage_stream_{key}": value
                    for key, value in series_moments(
                        estimand["monthly_series"]
                    ).items()
                },
            }
            for estimand in (primary, coprimary)
        ]
    )

    dsr_table.to_csv(OUT / "deflated_sharpe_multiple_testing.csv", index=False)
    inputs_table.to_csv(OUT / "deflated_sharpe_inputs.csv", index=False)
    haircut_table.to_csv(OUT / "deflated_sharpe_haircut.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print("\n=== Trial counts ===")
    print(f"N_structural (inherited-parameter argument): {N_STRUCTURAL}")
    print(f"N_reported (counted confirmatory p-values)  : {n_reported}")
    print(f"  family composition: {family['family'].value_counts().to_dict()}")
    print(f"N_literature (HLZ external anchor)          : {N_LITERATURE}")

    print("\n=== Deflated Sharpe inputs (advantage streams) ===")
    print(inputs_table.to_string(index=False))

    print("\n=== Deflated Sharpe Ratio over trial ladder ===")
    show = [
        "estimand",
        "trials_N",
        "expected_max_standard_normal",
        "t_observed_paper",
        "dsr_t_anchored",
        "implied_one_sided_p_after_N",
        "dsr_closed_form_advantage_stream",
    ]
    print(dsr_table[show].to_string(index=False))

    print("\n=== Harvey-Liu-Zhu haircut (one-sided p) ===")
    print(haircut_table.to_string(index=False))

    print(
        "\nWrote deflated_sharpe_multiple_testing.csv, "
        "deflated_sharpe_inputs.csv and deflated_sharpe_haircut.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
