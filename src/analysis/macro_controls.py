"""Run the registered macroeconomic controls on the thirteen-market panels.

Inputs: primary H1/H2 result CSV files and dated public macro-control CSV
snapshots.
Outputs: control panels, estimates, sensitivity files, audit inventory, and
French report in ``results/controls-13-markets-2026-07-27``.
Purpose: assess whether oil position, central-bank balance sheets, or broad
money account for the published H1/H2 results.
"""
from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from analysis import run_full_sample as R
from common.names import EURO, MARKETS
from common.paths import FULL_SAMPLE_RESULTS, MACRO_DATA, MACRO_RESULTS, REPO_ROOT

SOURCE_OUT = FULL_SAMPLE_RESULTS
OUT = MACRO_RESULTS
MACRO = MACRO_DATA
REPORT = OUT / "RAPPORT-CONTROLES-MACRO-13-MARCHES-FR.md"

ISO2 = {
    "AUS": "AU",
    "BEL": "BE",
    "CAN": "CA",
    "CHE": "CH",
    "DEU": "DE",
    "ESP": "ES",
    "FRA": "FR",
    "GBR": "GB",
    "JPN": "JP",
    "NLD": "NL",
    "NOR": "NO",
    "USA": "US",
    "ZAF": "ZA",
}
MONEY_SOURCE = {
    "AUS": "AUS",
    "BEL": "EA20",
    "CAN": "CAN",
    "CHE": "CHE",
    "DEU": "EA20",
    "ESP": "EA20",
    "FRA": "EA20",
    "GBR": "GBR",
    "JPN": "JPN",
    "NLD": "EA20",
    "NOR": "NOR",
    "USA": "USA",
    "ZAF": "ZAF",
}
CONTROL_LABELS = {
    "oil_score": "Position pétrolière nette, asinh",
    "petroleum_surplus": "Surplus pétrolier binaire",
    "balance_sheet_change_pp": "Expansion du bilan, points de PIB",
    "broad_money_growth_lag1": "Croissance de la monnaie large, retard 1 mois",
    "broad_money_growth_lag2": "Croissance de la monnaie large, retard 2 mois",
    "broad_money_growth_lag3": "Croissance de la monnaie large, retard 3 mois",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_h2_monthly_panel() -> pd.DataFrame:
    """Rebuild H2 returns and verify all 13 country-level Sharpe differences."""
    R.wire_everything()
    keys = R.energy_keys()
    expected = pd.read_csv(SOURCE_OUT / "h2_per_market.csv").set_index("market")
    rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for market in MARKETS:
        print(f"[H2 monthly, 13 markets] {market}", flush=True)
        energy, _ = (
            R.build_usa_energy()
            if market == "USA"
            else R.build_global_energy(market, keys)
        )
        result = R.run_h2_market(market, energy)
        currency = result["series"]["currency_only"]
        energy_rule = result["series"]["currency_energy"]
        bill = result["returns"]["bill"].reindex(currency.index)
        excess_currency = currency - bill
        excess_energy = energy_rule - bill
        z = (
            excess_energy / excess_energy.rolling(36).std().shift(1)
            - excess_currency / excess_currency.rolling(36).std().shift(1)
        )
        rows.append(
            pd.DataFrame(
                {
                    "market": market,
                    "month": currency.index.astype(str),
                    "d": (energy_rule - currency).to_numpy(float),
                    "z": z.to_numpy(float),
                    "return_currency_only": currency.to_numpy(float),
                    "return_currency_energy": energy_rule.to_numpy(float),
                }
            )
        )
        measured = (
            R.sharpe(energy_rule, result["returns"]["bill"])
            - R.sharpe(currency, result["returns"]["bill"])
        )
        reference = float(expected.loc[market, "augmentation_advantage"])
        error = abs(measured - reference)
        audit_rows.append(
            {
                "market": market,
                "computed_augmentation_advantage": measured,
                "reference_augmentation_advantage": reference,
                "absolute_error": error,
                "verdict": "PASS" if error <= 1e-12 else "FAIL",
            }
        )
    panel = pd.concat(rows, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    if set(panel["market"]) != set(MARKETS):
        raise RuntimeError("H2 panel does not contain exactly the 13 markets")
    if panel.duplicated(["market", "month"]).any():
        raise RuntimeError("Duplicate H2 market-month rows")
    if (audit["verdict"] != "PASS").any():
        raise RuntimeError("H2 monthly reconstruction failed")
    panel.to_csv(OUT / "h2_return_panel.csv", index=False)
    audit.to_csv(OUT / "h2_monthly_reconstruction_audit.csv", index=False)
    return panel


def oil_panel() -> pd.DataFrame:
    data = pd.read_csv(MACRO / "eia_petroleum_position_annual.csv")
    data = data[data["market"].isin(MARKETS)].copy()
    if set(data["market"]) != set(MARKETS):
        raise RuntimeError("EIA oil data do not contain all 13 markets")
    data["oil_score"] = np.arcsinh(data["net_petroleum_position"])
    data["outcome_year"] = data["year"] + 1
    return data[
        [
            "market",
            "outcome_year",
            "production_qbtu",
            "consumption_qbtu",
            "net_petroleum_position",
            "petroleum_surplus",
            "oil_score",
        ]
    ]


def balance_sheet_panel() -> pd.DataFrame:
    data = pd.read_csv(MACRO / "bis_central_bank_assets_pct_gdp_quarterly.csv")
    data["quarter_period"] = pd.PeriodIndex(data["quarter"], freq="Q")
    data = data.sort_values(["source_area", "quarter_period"])
    data["balance_sheet_change_pp"] = data.groupby("source_area")[
        "assets_pct_gdp"
    ].diff(4)
    data["outcome_quarter"] = data["quarter_period"] + 1
    rows = []
    for market in MARKETS:
        source_area = "XM" if market in EURO else ISO2[market]
        sample = data[data["source_area"] == source_area].copy()
        if sample.empty:
            raise RuntimeError(f"{market}: BIS source {source_area} is empty")
        sample["market"] = market
        sample["policy_source_area"] = source_area
        rows.append(
            sample[
                [
                    "market",
                    "outcome_quarter",
                    "assets_pct_gdp",
                    "balance_sheet_change_pp",
                    "policy_source_area",
                ]
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    complete = (
        result.dropna(subset=["balance_sheet_change_pp"])
        .groupby("outcome_quarter")["market"]
        .nunique()
    )
    complete = set(complete[complete == len(MARKETS)].index)
    result = result[result["outcome_quarter"].isin(complete)].copy()
    if set(result["market"]) != set(MARKETS):
        raise RuntimeError("The common BIS window does not contain all 13 markets")
    if result.duplicated(["market", "outcome_quarter"]).any():
        raise RuntimeError("Duplicate BIS market-quarter rows")
    return result


def broad_money_panel() -> pd.DataFrame:
    source = pd.read_csv(MACRO / "oecd_broad_money_monthly_sources.csv")
    source["reference_month"] = pd.PeriodIndex(source["month"], freq="M")
    source = source.sort_values(["source_area", "reference_month"])
    if not (source["broad_money_xdc"] > 0).all():
        raise RuntimeError("Broad-money levels must be strictly positive")
    source["broad_money_growth_yoy"] = 100 * source.groupby("source_area")[
        "broad_money_xdc"
    ].transform(lambda values: np.log(values).diff(12))

    mapped = []
    for market in MARKETS:
        source_area = MONEY_SOURCE[market]
        sample = source[source["source_area"] == source_area].copy()
        if market in EURO:
            sample = sample[sample["reference_month"] >= pd.Period("1999-01", "M")]
        sample["market"] = market
        mapped.append(
            sample[
                [
                    "market",
                    "source_area",
                    "reference_month",
                    "broad_money_xdc",
                    "broad_money_growth_yoy",
                ]
            ]
        )
    base = pd.concat(mapped, ignore_index=True)
    frames = []
    for lag in (1, 2, 3):
        sample = base.dropna(subset=["broad_money_growth_yoy"]).copy()
        sample["outcome_month"] = sample["reference_month"] + lag
        sample = sample.rename(
            columns={
                "source_area": f"money_source_area_lag{lag}",
                "reference_month": f"money_reference_month_lag{lag}",
                "broad_money_xdc": f"broad_money_xdc_lag{lag}",
                "broad_money_growth_yoy": f"broad_money_growth_lag{lag}",
            }
        )
        complete = sample.groupby("outcome_month")["market"].nunique()
        complete = set(complete[complete == len(MARKETS)].index)
        sample = sample[sample["outcome_month"].isin(complete)].copy()
        keep = [
            "market",
            "outcome_month",
            f"money_source_area_lag{lag}",
            f"money_reference_month_lag{lag}",
            f"broad_money_xdc_lag{lag}",
            f"broad_money_growth_lag{lag}",
        ]
        frames.append(sample[keep])
    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(
            frame,
            on=["market", "outcome_month"],
            how="outer",
            validate="one_to_one",
        )
    if set(result["market"]) != set(MARKETS):
        raise RuntimeError("Broad-money panel does not contain all 13 markets")
    if result.duplicated(["market", "outcome_month"]).any():
        raise RuntimeError("Duplicate broad-money market-month rows")
    return result


def attach_controls(panel: pd.DataFrame) -> pd.DataFrame:
    data = panel.copy()
    data["outcome_month"] = pd.PeriodIndex(data["month"], freq="M")
    data["outcome_year"] = data["outcome_month"].dt.year
    data["outcome_quarter"] = data["outcome_month"].dt.asfreq("Q")
    data = data.merge(
        oil_panel(),
        on=["market", "outcome_year"],
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        balance_sheet_panel(),
        on=["market", "outcome_quarter"],
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        broad_money_panel(),
        on=["market", "outcome_month"],
        how="left",
        validate="many_to_one",
    )
    return data


def panel_index(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["time"] = pd.PeriodIndex(result["month"], freq="M").to_timestamp()
    return result.set_index(["market", "time"]).sort_index()


def residualize(
    data: pd.DataFrame,
    variable: str,
    controls: list[str],
) -> pd.Series:
    indexed = panel_index(data)
    exog = (
        indexed[controls].astype(float)
        if controls
        else pd.DataFrame({"constant": 1.0}, index=indexed.index)
    )
    fit = PanelOLS(
        indexed[variable].astype(float),
        exog,
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(cov_type="unadjusted")
    return fit.resids


def exact_country_score_signflip_p(
    data: pd.DataFrame,
    dependent: str,
    target: str,
    other_controls: list[str],
) -> float:
    y_resid = residualize(data, dependent, other_controls)
    x_resid = residualize(data, target, other_controls)
    common = y_resid.index.intersection(x_resid.index)
    scores = (
        (y_resid.loc[common] * x_resid.loc[common])
        .groupby(level="market")
        .sum()
        .to_numpy(float)
    )
    observed = abs(float(scores.sum()))
    signs = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=len(scores))),
        dtype=float,
    )
    draws = np.abs(signs @ scores)
    return float(np.mean(draws >= observed - 1e-15))


def fit_panel(
    data: pd.DataFrame,
    dependent: str,
    regressors: list[str],
    model_name: str,
) -> list[dict[str, object]]:
    needed = [dependent, *regressors, "market", "month"]
    sample = data[needed].dropna().copy()
    if set(sample["market"]) != set(MARKETS):
        raise RuntimeError(
            f"{dependent}/{model_name}: expected 13 markets, got "
            f"{sorted(sample['market'].unique())}"
        )
    indexed = panel_index(sample)
    fit = PanelOLS(
        indexed[dependent].astype(float),
        indexed[regressors].astype(float),
        entity_effects=True,
        time_effects=True,
        drop_absorbed=True,
    ).fit(
        cov_type="clustered",
        cluster_entity=True,
        cluster_time=True,
        debiased=True,
    )
    rows = []
    for regressor in regressors:
        other = [value for value in regressors if value != regressor]
        p_value = float(fit.pvalues[regressor])
        rows.append(
            {
                "dependent_variable": dependent,
                "model": model_name,
                "regressor": regressor,
                "observations": len(sample),
                "markets": sample["market"].nunique(),
                "start": sample["month"].min(),
                "end": sample["month"].max(),
                "coefficient": float(fit.params[regressor]),
                "regressor_sd": float(sample[regressor].std(ddof=1)),
                "effect_per_one_sd": (
                    float(fit.params[regressor])
                    * float(sample[regressor].std(ddof=1))
                ),
                "two_way_cluster_se": float(fit.std_errors[regressor]),
                "two_way_cluster_t": float(fit.tstats[regressor]),
                "two_way_cluster_p_two_sided": p_value,
                "stars_two_way": stars(p_value),
                "country_score_signflip_p_two_sided": (
                    exact_country_score_signflip_p(
                        sample,
                        dependent,
                        regressor,
                        other,
                    )
                ),
                "country_fixed_effects": True,
                "calendar_fixed_effects": True,
                "other_regressors": " ".join(other) if other else "none",
            }
        )
    return rows


def stars(p_value: float) -> str:
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def holm(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    order = np.argsort(array)
    adjusted = np.empty(len(array))
    running = 0.0
    for rank, position in enumerate(order):
        running = max(running, min(1.0, (len(array) - rank) * array[position]))
        adjusted[position] = running
    return pd.Series(adjusted, index=values.index)


def run_main_models(
    h1_risk: pd.DataFrame,
    h1_raw: pd.DataFrame,
    h2: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    outcomes = {
        "h1_risk_adjusted": attach_controls(
            h1_risk.rename(columns={"z": "h1_risk_adjusted"})
        ),
        "h1_raw_return_difference": attach_controls(
            h1_raw.rename(columns={"d": "h1_raw_return_difference"})
        ),
        "h2_risk_adjusted": attach_controls(
            h2.rename(columns={"z": "h2_risk_adjusted"})
        ),
        "h2_raw_return_difference": attach_controls(
            h2.rename(columns={"d": "h2_raw_return_difference"})
        ),
    }
    specifications = [
        ("oil_continuous", ["oil_score"]),
        ("oil_surplus_binary", ["petroleum_surplus"]),
        ("balance_sheet", ["balance_sheet_change_pp"]),
        ("broad_money", ["broad_money_growth_lag2"]),
        (
            "joint",
            [
                "oil_score",
                "balance_sheet_change_pp",
                "broad_money_growth_lag2",
            ],
        ),
    ]
    rows: list[dict[str, object]] = []
    for dependent, data in outcomes.items():
        for model_name, regressors in specifications:
            rows.extend(fit_panel(data, dependent, regressors, model_name))
    result = pd.DataFrame(rows)
    result["core_test"] = (
        result["dependent_variable"].isin(
            ["h1_risk_adjusted", "h2_risk_adjusted"]
        )
        & (
            (
                result["model"].eq("oil_continuous")
                & result["regressor"].eq("oil_score")
            )
            | (
                result["model"].eq("balance_sheet")
                & result["regressor"].eq("balance_sheet_change_pp")
            )
            | (
                result["model"].eq("broad_money")
                & result["regressor"].eq("broad_money_growth_lag2")
            )
        )
    )
    result["two_way_cluster_holm_six"] = np.nan
    result["score_signflip_holm_six"] = np.nan
    core = result["core_test"]
    result.loc[core, "two_way_cluster_holm_six"] = holm(
        result.loc[core, "two_way_cluster_p_two_sided"]
    )
    result.loc[core, "score_signflip_holm_six"] = holm(
        result.loc[core, "country_score_signflip_p_two_sided"]
    )
    result.to_csv(OUT / "macro_control_panel_results.csv", index=False)
    return result, outcomes


def h1_alpha_control_comparison(
    h1_risk: pd.DataFrame,
) -> pd.DataFrame:
    """Separate the common-sample effect from the effect of macro controls.

    The central H1 alpha is a pooled constant without fixed effects.  This
    comparison therefore keeps the same pooled estimand.  The controlled
    column centers every macro regressor on the common sample so that its
    constant is the advantage at the sample-average macro state and remains
    directly comparable with the common-sample constant.  A fourth,
    diagnostic specification leaves the controls uncentered and reports the
    intercept at the economically special point where all three controls are
    zero.
    """
    dependent = "h1_risk_adjusted"
    controls = [
        "oil_score",
        "balance_sheet_change_pp",
        "broad_money_growth_lag2",
    ]
    full = h1_risk.rename(columns={"z": dependent})[
        ["market", "month", dependent]
    ].dropna().copy()
    attached = attach_controls(
        h1_risk.rename(columns={"z": dependent})
    )
    common = attached[
        ["market", "month", dependent, *controls]
    ].dropna().copy()
    if set(full["market"]) != set(MARKETS):
        raise RuntimeError("Full H1 alpha sample does not contain 13 markets")
    if set(common["market"]) != set(MARKETS):
        raise RuntimeError("Common macro sample does not contain 13 markets")
    if common.duplicated(["market", "month"]).any():
        raise RuntimeError("Duplicate rows in common H1 macro sample")

    centered_controls = []
    for control in controls:
        centered = f"{control}_centered"
        common[centered] = common[control] - common[control].mean()
        centered_controls.append(centered)

    def sample_digest(data: pd.DataFrame) -> str:
        keys = (
            data[["market", "month"]]
            .sort_values(["market", "month"])
            .astype(str)
            .agg("|".join, axis=1)
        )
        return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()

    full_same_calendar_window = full[
        full["month"].between(common["month"].min(), common["month"].max())
    ].copy()
    common_equals_calendar_window = (
        sample_digest(full_same_calendar_window) == sample_digest(common)
    )
    if not common_equals_calendar_window:
        raise RuntimeError(
            "Common macro rows do not equal the complete H1 calendar window"
        )

    specifications = [
        {
            "column": "1",
            "specification": "base_full_sample",
            "data": full,
            "regressors": [],
            "controls": "none",
            "control_reference": "not_applicable",
        },
        {
            "column": "2",
            "specification": "base_common_macro_sample",
            "data": common,
            "regressors": [],
            "controls": "none",
            "control_reference": "not_applicable",
        },
        {
            "column": "3",
            "specification": "controls_common_sample_centered",
            "data": common,
            "regressors": centered_controls,
            "controls": "oil balance_sheet broad_money",
            "control_reference": "common_sample_mean",
        },
        {
            "column": "diagnostic",
            "specification": "controls_common_sample_zero_reference",
            "data": common,
            "regressors": controls,
            "controls": "oil balance_sheet broad_money",
            "control_reference": "all_controls_equal_zero",
        },
    ]
    covariance_specs = [
        (
            "calendar_month_cluster",
            {"cov_type": "clustered", "cluster_time": True},
        ),
        (
            "country_calendar_two_way",
            {
                "cov_type": "clustered",
                "cluster_entity": True,
                "cluster_time": True,
            },
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
    rows: list[dict[str, object]] = []
    for specification in specifications:
        data = specification["data"]
        regressors = specification["regressors"]
        indexed = panel_index(data)
        exog = pd.DataFrame({"constant": 1.0}, index=indexed.index)
        for regressor in regressors:
            exog[regressor] = indexed[regressor].astype(float)
        model = PanelOLS(
            indexed[dependent].astype(float),
            exog,
        )
        for covariance_name, keywords in covariance_specs:
            fit = model.fit(debiased=True, **keywords)
            alpha_monthly = float(fit.params["constant"])
            standard_error_monthly = float(fit.std_errors["constant"])
            t_statistic = alpha_monthly / standard_error_monthly
            row = {
                "column": specification["column"],
                "specification": specification["specification"],
                "covariance": covariance_name,
                "observations": len(data),
                "markets": data["market"].nunique(),
                "start": data["month"].min(),
                "end": data["month"].max(),
                "sample_key_sha256": sample_digest(data),
                "common_sample_equals_complete_h1_calendar_window": (
                    common_equals_calendar_window
                ),
                "controls": specification["controls"],
                "control_reference": specification["control_reference"],
                "country_fixed_effects": False,
                "calendar_fixed_effects": False,
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
            for control in controls:
                term = (
                    f"{control}_centered"
                    if specification["control_reference"]
                    == "common_sample_mean"
                    else control
                )
                row[f"{control}_coefficient"] = (
                    float(fit.params[term])
                    if term in fit.params.index
                    else np.nan
                )
            rows.append(row)
    result = pd.DataFrame(rows)

    common_hashes = result[
        result["column"].isin(["2", "3", "diagnostic"])
    ]["sample_key_sha256"].unique()
    if len(common_hashes) != 1:
        raise RuntimeError("Controlled H1 columns do not use the same rows")

    existing = pd.read_csv(
        SOURCE_OUT / "h1_joint_dependence_inference.csv"
    ).set_index("method")
    first = result[result["column"].eq("1")].set_index("covariance")
    for method in covariance_specs:
        name = method[0]
        for new_column, old_column in [
            ("alpha_annualized_sharpe", "alpha_annualized_sharpe"),
            ("standard_error_annualized", "standard_error_annualized"),
            ("t_statistic", "t_statistic"),
            ("p_one_sided_normal", "p_one_sided_normal"),
        ]:
            error = abs(
                float(first.loc[name, new_column])
                - float(existing.loc[name, old_column])
            )
            if error > 1e-12:
                raise RuntimeError(
                    f"Full-sample H1 reconciliation failed for "
                    f"{name}/{new_column}: {error}"
                )

    result.to_csv(
        OUT / "h1_alpha_controls_comparison_2026-07-27.csv",
        index=False,
    )
    return result


def covariance_and_influence(
    outcomes: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    covariance_rows: list[dict[str, object]] = []
    influence_rows: list[dict[str, object]] = []
    specifications = [
        ("oil_continuous", "oil_score"),
        ("oil_surplus_binary", "petroleum_surplus"),
        ("balance_sheet", "balance_sheet_change_pp"),
        ("broad_money", "broad_money_growth_lag2"),
    ]
    covariance_specs = [
        ("cluster_country", {"cov_type": "clustered", "cluster_entity": True}),
        ("cluster_calendar", {"cov_type": "clustered", "cluster_time": True}),
        (
            "cluster_country_calendar",
            {
                "cov_type": "clustered",
                "cluster_entity": True,
                "cluster_time": True,
            },
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
    for dependent in ("h1_risk_adjusted", "h2_risk_adjusted"):
        data = outcomes[dependent]
        for model_name, regressor in specifications:
            sample = data[
                [dependent, regressor, "market", "month"]
            ].dropna().copy()
            indexed = panel_index(sample)
            model = PanelOLS(
                indexed[dependent].astype(float),
                indexed[[regressor]].astype(float),
                entity_effects=True,
                time_effects=True,
            )
            for covariance_name, keywords in covariance_specs:
                fit = model.fit(debiased=True, **keywords)
                p_value = float(fit.pvalues[regressor])
                covariance_rows.append(
                    {
                        "dependent_variable": dependent,
                        "model": model_name,
                        "regressor": regressor,
                        "covariance": covariance_name,
                        "observations": len(sample),
                        "markets": sample["market"].nunique(),
                        "coefficient": float(fit.params[regressor]),
                        "standard_error": float(fit.std_errors[regressor]),
                        "t": float(fit.tstats[regressor]),
                        "p_two_sided": p_value,
                        "stars": stars(p_value),
                    }
                )
            for omitted in sorted(sample["market"].unique()):
                reduced = sample[sample["market"] != omitted]
                reduced_indexed = panel_index(reduced)
                fit = PanelOLS(
                    reduced_indexed[dependent].astype(float),
                    reduced_indexed[[regressor]].astype(float),
                    entity_effects=True,
                    time_effects=True,
                ).fit(
                    cov_type="clustered",
                    cluster_entity=True,
                    cluster_time=True,
                    debiased=True,
                )
                influence_rows.append(
                    {
                        "dependent_variable": dependent,
                        "model": model_name,
                        "regressor": regressor,
                        "omitted_market": omitted,
                        "observations": len(reduced),
                        "markets": reduced["market"].nunique(),
                        "coefficient": float(fit.params[regressor]),
                        "two_way_cluster_t": float(fit.tstats[regressor]),
                        "two_way_cluster_p_two_sided": float(
                            fit.pvalues[regressor]
                        ),
                    }
                )
    covariance = pd.DataFrame(covariance_rows)
    influence = pd.DataFrame(influence_rows)
    covariance.to_csv(OUT / "macro_control_covariance_sensitivity.csv", index=False)
    influence.to_csv(OUT / "macro_control_leave_one_out.csv", index=False)
    return covariance, influence


def money_lag_sensitivity(outcomes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dependent in ("h1_risk_adjusted", "h2_risk_adjusted"):
        for lag in (1, 2, 3):
            regressor = f"broad_money_growth_lag{lag}"
            rows.extend(
                fit_panel(
                    outcomes[dependent],
                    dependent,
                    [regressor],
                    f"broad_money_lag_{lag}",
                )
            )
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "broad_money_lag_sensitivity.csv", index=False)
    return result


def coverage_table(outcomes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dependent in ("h1_risk_adjusted", "h2_risk_adjusted"):
        data = outcomes[dependent]
        for model_name, controls in [
            ("oil_continuous", ["oil_score"]),
            ("balance_sheet", ["balance_sheet_change_pp"]),
            ("broad_money", ["broad_money_growth_lag2"]),
            (
                "joint",
                [
                    "oil_score",
                    "balance_sheet_change_pp",
                    "broad_money_growth_lag2",
                ],
            ),
        ]:
            sample = data.dropna(subset=[dependent, *controls]).copy()
            counts = sample.groupby("market").size()
            rows.append(
                {
                    "dependent_variable": dependent,
                    "model": model_name,
                    "controls": " ".join(controls),
                    "observations": len(sample),
                    "markets": sample["market"].nunique(),
                    "start": sample["month"].min(),
                    "end": sample["month"].max(),
                    "minimum_country_observations": int(counts.min()),
                    "maximum_country_observations": int(counts.max()),
                }
            )
    result = pd.DataFrame(rows)
    if not (result["markets"] == len(MARKETS)).all():
        raise RuntimeError("At least one main control specification loses a market")
    result.to_csv(OUT / "macro_control_coverage.csv", index=False)
    return result


def result_row(
    results: pd.DataFrame,
    dependent: str,
    model: str,
    regressor: str,
) -> pd.Series:
    sample = results[
        (results["dependent_variable"] == dependent)
        & (results["model"] == model)
        & (results["regressor"] == regressor)
    ]
    if len(sample) != 1:
        raise RuntimeError(
            f"Result is not unique: {dependent}/{model}/{regressor}"
        )
    return sample.iloc[0]


def markdown_result_table(results: pd.DataFrame) -> str:
    rows = [
        "| Variable dépendante | Variable explicative | Coefficient | Erreur standard | "
        "Statistique t | Valeur p | Holm p | Obs. | Pays | Effets fixes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dependent in ("h1_risk_adjusted", "h2_risk_adjusted"):
        for model, regressor in [
            ("oil_continuous", "oil_score"),
            ("balance_sheet", "balance_sheet_change_pp"),
            ("broad_money", "broad_money_growth_lag2"),
        ]:
            row = result_row(results, dependent, model, regressor)
            star = (
                row.stars_two_way
                if isinstance(row.stars_two_way, str)
                else ""
            )
            rows.append(
                f"| {dependent} | {CONTROL_LABELS[regressor]} | "
                f"{row.coefficient:+.6f}{star} | "
                f"{row.two_way_cluster_se:.6f} | "
                f"{row.two_way_cluster_t:+.3f} | "
                f"{row.two_way_cluster_p_two_sided:.4f} | "
                f"{row.two_way_cluster_holm_six:.4f} | "
                f"{int(row.observations)} | {int(row.markets)} | pays et calendrier |"
            )
    return "\n".join(rows)


def write_report(
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    covariance: pd.DataFrame,
    influence: pd.DataFrame,
    lag_results: pd.DataFrame,
    alpha_comparison: pd.DataFrame,
) -> None:
    h1_joint = pd.read_csv(
        SOURCE_OUT / "h1_joint_dependence_inference.csv"
    )

    def h1_joint_row(method: str) -> pd.Series:
        sample = h1_joint[h1_joint["method"] == method]
        if len(sample) != 1:
            raise RuntimeError(f"H1 joint inference row missing: {method}")
        return sample.iloc[0]

    h1_calendar = h1_joint_row("calendar_month_cluster")
    h1_two_way = h1_joint_row("country_calendar_two_way")
    h1_dk12 = h1_joint_row("driscoll_kraay_12")
    h1_dk24 = h1_joint_row("driscoll_kraay_24")

    def alpha_row(column: str, covariance_name: str) -> pd.Series:
        sample = alpha_comparison[
            alpha_comparison["column"].astype(str).eq(column)
            & alpha_comparison["covariance"].eq(covariance_name)
        ]
        if len(sample) != 1:
            raise RuntimeError(
                f"H1 alpha comparison row missing: "
                f"{column}/{covariance_name}"
            )
        return sample.iloc[0]

    alpha_1_calendar = alpha_row("1", "calendar_month_cluster")
    alpha_2_calendar = alpha_row("2", "calendar_month_cluster")
    alpha_3_calendar = alpha_row("3", "calendar_month_cluster")
    alpha_1_dk12 = alpha_row("1", "driscoll_kraay_12")
    alpha_2_dk12 = alpha_row("2", "driscoll_kraay_12")
    alpha_3_dk12 = alpha_row("3", "driscoll_kraay_12")
    alpha_zero_calendar = alpha_row(
        "diagnostic", "calendar_month_cluster"
    )
    alpha_zero_dk12 = alpha_row("diagnostic", "driscoll_kraay_12")

    def describe(dependent: str, model: str, regressor: str) -> str:
        row = result_row(results, dependent, model, regressor)
        direction = "positif" if row.coefficient > 0 else "négatif"
        level = (
            "au seuil de 1 pour cent"
            if row.two_way_cluster_p_two_sided < 0.01
            else "au seuil de 5 pour cent"
            if row.two_way_cluster_p_two_sided < 0.05
            else "au seuil de 10 pour cent"
            if row.two_way_cluster_p_two_sided < 0.10
            else "à aucun des seuils de 10, 5 ou 1 pour cent"
        )
        return (
            f"coefficient {direction} de {row.coefficient:+.6f}, "
            f"t={row.two_way_cluster_t:+.3f}, "
            f"p={row.two_way_cluster_p_two_sided:.4f}, {level}, "
            f"Holm p={row.two_way_cluster_holm_six:.4f}"
        )

    def covariance_p(dependent: str, regressor: str, name: str) -> float:
        sample = covariance[
            (covariance["dependent_variable"] == dependent)
            & (covariance["regressor"] == regressor)
            & (covariance["covariance"] == name)
        ]
        return float(sample.iloc[0]["p_two_sided"])

    def loo_summary(dependent: str, regressor: str) -> str:
        sample = influence[
            (influence["dependent_variable"] == dependent)
            & (influence["regressor"] == regressor)
        ]
        same_sign = (
            (sample["coefficient"] > 0).all()
            or (sample["coefficient"] < 0).all()
        )
        return (
            f"[{sample['coefficient'].min():+.6f}, "
            f"{sample['coefficient'].max():+.6f}], "
            f"signe {'stable' if same_sign else 'instable'}"
        )

    h1_oil_binary = result_row(
        results,
        "h1_risk_adjusted",
        "oil_surplus_binary",
        "petroleum_surplus",
    )
    h1_raw_oil_binary = result_row(
        results,
        "h1_raw_return_difference",
        "oil_surplus_binary",
        "petroleum_surplus",
    )
    eia = pd.read_csv(MACRO / "eia_petroleum_position_annual.csv")
    surplus_markets = sorted(
        eia[
            eia["market"].isin(MARKETS)
            & eia["petroleum_surplus"].eq(1)
        ]["market"].unique()
    )

    lag_table_rows = [
        "| Variable dépendante | Retard | Coefficient | t | Valeur p | Obs. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in lag_results.itertuples():
        lag = int(row.regressor.rsplit("lag", 1)[1])
        star = (
            row.stars_two_way
            if isinstance(row.stars_two_way, str)
            else ""
        )
        lag_table_rows.append(
            f"| {row.dependent_variable} | {lag} mois | "
            f"{row.coefficient:+.6f}{star} | "
            f"{row.two_way_cluster_t:+.3f} | "
            f"{row.two_way_cluster_p_two_sided:.4f} | "
            f"{int(row.observations)} |"
        )

    coverage_lines = [
        "| Variable dépendante | Modèle | Observations | Pays | Début | Fin | "
        "Minimum par pays | Maximum par pays |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]
    for row in coverage.itertuples():
        coverage_lines.append(
            f"| {row.dependent_variable} | {row.model} | "
            f"{int(row.observations)} | {int(row.markets)} | "
            f"{row.start} | {row.end} | "
            f"{int(row.minimum_country_observations)} | "
            f"{int(row.maximum_country_observations)} |"
        )

    text = f"""# Contrôles macroéconomiques sur les treize marchés

Date d'exécution : 27 juillet 2026

## Statut

Cette analyse est exploratoire et postérieure aux résultats principaux. Elle utilise uniquement
les treize marchés du papier. Elle ne modifie pas les portefeuilles, les résultats H1 et H2 sans
contrôle, ni le papier français au moment de son exécution.

La position pétrolière, l'expansion du bilan et la croissance de la monnaie au sens large sont
trois variables différentes. La variable de bilan ne doit pas être appelée masse monétaire.

## Inférence jointe sur la constante H1

La constante H1 annualisée reste {h1_calendar.alpha_annualized_sharpe:+.6f}. Avec le groupement
par mois calendaire, t={h1_calendar.t_statistic:+.3f} et la valeur p unilatérale est
{h1_calendar.p_one_sided_normal:.4f}. Le groupement à deux dimensions, pays et calendrier, donne
t={h1_two_way.t_statistic:+.3f} et p={h1_two_way.p_one_sided_normal:.4f}.

Driscoll et Kraay donne t={h1_dk12.t_statistic:+.3f} et
p={h1_dk12.p_one_sided_normal:.4f} à 12 retards, puis
t={h1_dk24.t_statistic:+.3f} et p={h1_dk24.p_one_sided_normal:.4f} à 24 retards. Le seuil
unilatéral de 5 pour cent ne survit donc pas à l'inférence qui traite simultanément la dépendance
entre marchés et la dépendance sérielle. H1 reste significatif à 10 pour cent dans ces deux
spécifications.

## Alpha H1 sans et avec contrôles

| Ligne | (1) Base, histoire complète | (2) Base, lignes macro communes | (3) Contrôles, mêmes lignes |
|---|---:|---:|---:|
| Alpha annualisé | {alpha_1_calendar.alpha_annualized_sharpe:+.6f} | {alpha_2_calendar.alpha_annualized_sharpe:+.6f} | {alpha_3_calendar.alpha_annualized_sharpe:+.6f} |
| Erreur standard, mois groupés | {alpha_1_calendar.standard_error_annualized:.6f} | {alpha_2_calendar.standard_error_annualized:.6f} | {alpha_3_calendar.standard_error_annualized:.6f} |
| Valeur p unilatérale, mois groupés | {alpha_1_calendar.p_one_sided_normal:.4f} | {alpha_2_calendar.p_one_sided_normal:.4f} | {alpha_3_calendar.p_one_sided_normal:.4f} |
| Valeur p unilatérale, Driscoll et Kraay 12 | {alpha_1_dk12.p_one_sided_normal:.4f} | {alpha_2_dk12.p_one_sided_normal:.4f} | {alpha_3_dk12.p_one_sided_normal:.4f} |
| Observations | {int(alpha_1_calendar.observations)} | {int(alpha_2_calendar.observations)} | {int(alpha_3_calendar.observations)} |
| Période | {alpha_1_calendar.start} à {alpha_1_calendar.end} | {alpha_2_calendar.start} à {alpha_2_calendar.end} | {alpha_3_calendar.start} à {alpha_3_calendar.end} |
| Contrôles | aucun | aucun | pétrole, bilan, monnaie large |

Les colonnes (2) et (3) emploient exactement les mêmes
{int(alpha_2_calendar.observations)} lignes, dont l'empreinte est
`{alpha_2_calendar.sample_key_sha256}`. Dans la colonne (3), les contrôles sont centrés sur leur
moyenne dans cet échantillon. Ces lignes épuisent aussi toutes les observations H1 disponibles
entre janvier 2000 et décembre 2025~: aucune observation H1 n'est éliminée à l'intérieur de cette
fenêtre faute de contrôle. La constante mesure donc l'avantage au point macroéconomique
moyen et reste directement comparable à la colonne (2). Cette normalisation implique que les
deux constantes ponctuelles sont identiques ; l'effet éventuel des contrôles porte sur leur
précision et sur les pentes.

Le résultat attendu d'un alpha proche de +0,086 sur la fenêtre macro n'est pas confirmé. La
constante passe de {alpha_1_calendar.alpha_annualized_sharpe:+.6f} sur l'histoire complète à
{alpha_2_calendar.alpha_annualized_sharpe:+.6f} sur les lignes communes de 2000 à 2025, avant
même l'ajout des contrôles. Les valeurs p unilatérales Driscoll et Kraay sont respectivement
{alpha_1_dk12.p_one_sided_normal:.4f}, {alpha_2_dk12.p_one_sided_normal:.4f} et
{alpha_3_dk12.p_one_sided_normal:.4f}. Le changement vient donc du raccourcissement de
l'échantillon, pas des contrôles et pas du choix entre groupement calendaire et Driscoll et Kraay.

À titre de diagnostic, si les contrôles ne sont pas centrés, la constante décrit le point particulier
où la position pétrolière, l'expansion du bilan et la croissance monétaire sont toutes nulles. Elle
vaut alors {alpha_zero_calendar.alpha_annualized_sharpe:+.6f}, avec
p={alpha_zero_calendar.p_one_sided_normal:.4f} sous groupement calendaire et
p={alpha_zero_dk12.p_one_sided_normal:.4f} sous Driscoll et Kraay. Cette quantité n'est pas
l'avantage moyen de la fenêtre et n'est donc pas utilisée dans la comparaison principale.

## Couverture

{chr(10).join(coverage_lines)}

## Résultats centraux

Les étoiles utilisent les valeurs p bilatérales avec regroupement par pays et calendrier :
`*` pour 10 pour cent, `**` pour 5 pour cent et `***` pour 1 pour cent. La colonne Holm corrige
les six tests centraux.

{markdown_result_table(results)}

## Lecture de H1

* Pétrole : {describe("h1_risk_adjusted", "oil_continuous", "oil_score")}.
* Bilan de banque centrale : {describe("h1_risk_adjusted", "balance_sheet", "balance_sheet_change_pp")}.
* Monnaie au sens large : {describe("h1_risk_adjusted", "broad_money", "broad_money_growth_lag2")}.

La variante binaire du pétrole ne doit pas être omise. L'indicatrice de surplus donne un
coefficient de {h1_oil_binary.coefficient:+.6f}, t={h1_oil_binary.two_way_cluster_t:+.3f} et
p={h1_oil_binary.two_way_cluster_p_two_sided:.4f} avec le groupement pays-calendrier. Elle est
donc significative à 1 pour cent dans cette procédure. Le diagnostic ne se généralise cependant
pas : Driscoll et Kraay donne
p={covariance_p("h1_risk_adjusted", "petroleum_surplus", "driscoll_kraay_12"):.4f}
à 12 mois et
p={covariance_p("h1_risk_adjusted", "petroleum_surplus", "driscoll_kraay_24"):.4f}
à 24 mois, l'inversion de signes par pays donne
p={h1_oil_binary.country_score_signflip_p_two_sided:.4f}, et la différence brute de rendement
donne p={h1_raw_oil_binary.two_way_cluster_p_two_sided:.4f}. Le coefficient leave-one-out est
{loo_summary("h1_risk_adjusted", "petroleum_surplus")}. L'indicatrice est en outre parcimonieuse :
seuls {", ".join(surplus_markets)} connaissent au moins une année positive dans la source EIA.
Ce résultat est rapporté comme une sensibilité instable, et non retenu comme test pétrolier
central.

Pour la monnaie au sens large, Driscoll et Kraay donne
p={covariance_p("h1_risk_adjusted", "broad_money_growth_lag2", "driscoll_kraay_12"):.4f}
avec 12 mois et
p={covariance_p("h1_risk_adjusted", "broad_money_growth_lag2", "driscoll_kraay_24"):.4f}
avec 24 mois. Le coefficient après retrait successif d'un pays est
{loo_summary("h1_risk_adjusted", "broad_money_growth_lag2")}.

Pour l'expansion du bilan, les valeurs correspondantes de Driscoll et Kraay sont
p={covariance_p("h1_risk_adjusted", "balance_sheet_change_pp", "driscoll_kraay_12"):.4f}
et
p={covariance_p("h1_risk_adjusted", "balance_sheet_change_pp", "driscoll_kraay_24"):.4f}.
Le retrait pays par pays donne
{loo_summary("h1_risk_adjusted", "balance_sheet_change_pp")}.

## Lecture de H2

* Pétrole : {describe("h2_risk_adjusted", "oil_continuous", "oil_score")}.
* Bilan de banque centrale : {describe("h2_risk_adjusted", "balance_sheet", "balance_sheet_change_pp")}.
* Monnaie au sens large : {describe("h2_risk_adjusted", "broad_money", "broad_money_growth_lag2")}.

## Sensibilité au retard de la monnaie

{chr(10).join(lag_table_rows)}

## Interprétation

Un coefficient négatif signifie seulement que l'avantage mensuel ajusté du risque est plus faible
pendant les périodes où le contrôle augmente. Il s'agit d'une association conditionnelle avec
effets fixes, pas d'un effet causal.

La croissance de la monnaie au sens large est calculée à partir du stock MABM de l'OCDE. Elle est
retardée de deux mois dans la spécification centrale. Pour la Belgique, l'Allemagne, l'Espagne, la
France et les Pays-Bas, l'agrégat EA20 n'est utilisé qu'à partir de 1999.

Les actifs de banque centrale comprennent les réserves de change, les prêts d'urgence et d'autres
opérations. Une association avec cette série ne constitue pas une identification du QE.

## Décision de rédaction

L'audit machine, les fenêtres, les sensibilités de covariance et l'inventaire d'empreintes ont été
vérifiés. Ces résultats peuvent donc entrer dans la section de robustesse du papier, sous le statut
« analyse exploratoire de mécanisme ». Le texte ne doit pas présenter le coefficient monétaire
comme causal ni comme robuste à toutes les procédures : il est négatif dans H1, mais sa
significativité dépend du retard et de la covariance, et il ne survit pas à Holm dans les six tests
centraux.
"""
    REPORT.write_text(text, encoding="utf-8")


def write_audit(
    h1_risk: pd.DataFrame,
    h1_raw: pd.DataFrame,
    h2: pd.DataFrame,
    results: pd.DataFrame,
    coverage: pd.DataFrame,
    alpha_comparison: pd.DataFrame,
) -> None:
    common_alpha = alpha_comparison[
        alpha_comparison["column"].astype(str).isin(
            ["2", "3", "diagnostic"]
        )
    ]
    checks = [
        {
            "check": "H1 risk markets",
            "observed": h1_risk["market"].nunique(),
            "expected": len(MARKETS),
            "verdict": (
                "PASS"
                if set(h1_risk["market"]) == set(MARKETS)
                else "FAIL"
            ),
        },
        {
            "check": "H1 raw markets",
            "observed": h1_raw["market"].nunique(),
            "expected": len(MARKETS),
            "verdict": (
                "PASS"
                if set(h1_raw["market"]) == set(MARKETS)
                else "FAIL"
            ),
        },
        {
            "check": "H2 markets",
            "observed": h2["market"].nunique(),
            "expected": len(MARKETS),
            "verdict": (
                "PASS" if set(h2["market"]) == set(MARKETS) else "FAIL"
            ),
        },
        {
            "check": "Core tests",
            "observed": int(results["core_test"].sum()),
            "expected": 6,
            "verdict": "PASS" if int(results["core_test"].sum()) == 6 else "FAIL",
        },
        {
            "check": "Coverage rows with 13 markets",
            "observed": int((coverage["markets"] == len(MARKETS)).sum()),
            "expected": len(coverage),
            "verdict": (
                "PASS"
                if (coverage["markets"] == len(MARKETS)).all()
                else "FAIL"
            ),
        },
        {
            "check": "H1 alpha common-sample hashes",
            "observed": common_alpha["sample_key_sha256"].nunique(),
            "expected": 1,
            "verdict": (
                "PASS"
                if common_alpha["sample_key_sha256"].nunique() == 1
                else "FAIL"
            ),
        },
        {
            "check": "H1 alpha common-sample observations",
            "observed": common_alpha["observations"].nunique(),
            "expected": 1,
            "verdict": (
                "PASS"
                if common_alpha["observations"].nunique() == 1
                else "FAIL"
            ),
        },
        {
            "check": "H1 macro rows equal complete 2000-2025 H1 window",
            "observed": int(
                common_alpha[
                    "common_sample_equals_complete_h1_calendar_window"
                ].all()
            ),
            "expected": 1,
            "verdict": (
                "PASS"
                if common_alpha[
                    "common_sample_equals_complete_h1_calendar_window"
                ].all()
                else "FAIL"
            ),
        },
    ]
    audit = pd.DataFrame(checks)
    audit.to_csv(OUT / "macro_control_audit.csv", index=False)
    if (audit["verdict"] != "PASS").any():
        raise RuntimeError("Macro-control audit failed")


def output_inventory() -> None:
    names = [
        "h2_return_panel.csv",
        "h2_monthly_reconstruction_audit.csv",
        "macro_control_panel_results.csv",
        "macro_control_covariance_sensitivity.csv",
        "macro_control_leave_one_out.csv",
        "broad_money_lag_sensitivity.csv",
        "macro_control_coverage.csv",
        "macro_control_audit.csv",
        "h1_alpha_controls_comparison_2026-07-27.csv",
        "AUDIT-P0-H1-PETROLE-BINAIRE-FR.md",
        REPORT.name,
    ]
    rows = []
    for name in names:
        path = OUT / name
        rows.append(
            {
                "file": name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    for path in [
        SOURCE_OUT / "h1_risk_adjusted_panel.csv",
        SOURCE_OUT / "h1_return_panel.csv",
        SOURCE_OUT / "h2_per_market.csv",
        SOURCE_OUT / "h1_joint_dependence_inference.csv",
        MACRO / "eia_petroleum_position_annual.csv",
        MACRO / "bis_central_bank_assets_pct_gdp_quarterly.csv",
        MACRO / "oecd_broad_money_monthly_sources.csv",
        MACRO / "oecd_broad_money_source_manifest.json",
    ]:
        if path.parent == MACRO:
            logical_path = (
                f"data/macro_controls/2026-07-27/{path.name}"
            )
        else:
            logical_path = str(path.relative_to(REPO_ROOT))
        rows.append(
            {
                "file": logical_path,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "output_inventory_sha256.csv", index=False)


def main() -> int:
    required = [
        SOURCE_OUT / "h1_risk_adjusted_panel.csv",
        SOURCE_OUT / "h1_return_panel.csv",
        SOURCE_OUT / "h2_per_market.csv",
        SOURCE_OUT / "h1_joint_dependence_inference.csv",
        MACRO / "eia_petroleum_position_annual.csv",
        MACRO / "bis_central_bank_assets_pct_gdp_quarterly.csv",
        MACRO / "oecd_broad_money_monthly_sources.csv",
        MACRO / "oecd_broad_money_source_manifest.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.mkdir(parents=True, exist_ok=True)

    h2 = build_h2_monthly_panel()
    h1_risk = pd.read_csv(SOURCE_OUT / "h1_risk_adjusted_panel.csv")
    h1_raw = pd.read_csv(SOURCE_OUT / "h1_return_panel.csv")
    results, outcomes = run_main_models(h1_risk, h1_raw, h2)
    covariance, influence = covariance_and_influence(outcomes)
    lag_results = money_lag_sensitivity(outcomes)
    coverage = coverage_table(outcomes)
    alpha_comparison = h1_alpha_control_comparison(h1_risk)
    write_report(
        results,
        coverage,
        covariance,
        influence,
        lag_results,
        alpha_comparison,
    )
    write_audit(
        h1_risk,
        h1_raw,
        h2,
        results,
        coverage,
        alpha_comparison,
    )
    output_inventory()

    core = results[results["core_test"]][
        [
            "dependent_variable",
            "regressor",
            "coefficient",
            "two_way_cluster_t",
            "two_way_cluster_p_two_sided",
            "two_way_cluster_holm_six",
            "country_score_signflip_p_two_sided",
            "score_signflip_holm_six",
        ]
    ]
    print(core.to_json(orient="records", indent=2))
    print(REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
