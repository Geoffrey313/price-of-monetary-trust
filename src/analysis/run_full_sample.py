"""Reproduce H1--H3 on the harmonised thirteen-market universe.

Inputs: public snapshots and licensed reconstructed parquet inputs from the
configured data directory.
Outputs: the complete CSV, JSON, and PNG result set in
``results/full-sample``.
Purpose: run the paper's primary H1 monetary test, H2 energy test, H3 functional
form comparison, inference, robustness checks, and primary figures.

Every country-level family uses:

    AUS, BEL, CAN, FRA, DEU, JPN, NLD, NOR, ZAF, ESP, CHE, GBR, USA

The H2 energy sleeve is constructed whenever at least one listed energy firm
has a valid lagged market capitalisation.  The same rule is applied to every
market.  Concentration is reported rather than used as an exclusion rule.

Outputs are written below:
    docs/four_quadrant/results/full-sample/
"""
from __future__ import annotations

import itertools
import json
import os
from math import erf, sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binomtest, spearmanr

from analysis import euro_experiment as EU
from analysis.h3_variant import (
    lw_test,
    pooled_calendar_block_bootstrap,
    pooled_calendar_hac,
    run_variant,
    series_for,
)
from common.names import EURO, MARKETS, NAMES
from common.paths import DERIVED_ENERGY_DATA, FULL_SAMPLE_RESULTS, RECON_DATA, REPO_ROOT
from data.treasury_total_return import construct_monthly_tr
from engine import engine as E
from engine import market_wiring as P
from figures.i18n import L, figpath, market_name
from figures.plot_style import BLUE, GOLD, GREY, GRID, INK, ORANGE, RED

RECON = RECON_DATA
OUT = FULL_SAMPLE_RESULTS

# Currency conversion mirrors each market's reconstructed all-share index.
CUR = {
    "AUS": {"AUD": 1.0},
    "BEL": {"BEF": 40.3399, "EUR": 1.0},
    "CAN": {"CAD": 1.0},
    "CHE": {"CHF": 1.0},
    "DEU": {"DEM": 1.95583, "EUR": 1.0},
    "ESP": {"ESP": 166.386, "EUR": 1.0},
    "FRA": {"FRF": 6.55957, "EUR": 1.0},
    "GBR": {"GBP": 1.0},
    "JPN": {"JPY": 1.0},
    "NLD": {"NLG": 2.20371, "EUR": 1.0},
    "NOR": {"NOK": 1.0},
    "ZAF": {"ZAR": 1.0},
}

# World Bank API snapshot checked 2026-07-26, source last updated
# 2026-07-13.  2021 is the latest common non-missing year for all 13 markets
# and both indicators.
EXPOSURE_2021 = {
    "AUS": (4.10, 27.7060716965290),
    "BEL": (3.86, 8.81647347599391),
    "CAN": (6.55, 23.8688657837555),
    "CHE": (1.53, 1.12619429120116),
    "DEU": (2.70, 2.60156267352484),
    "ESP": (2.66, 6.00550281136412),
    "FRA": (3.23, 3.67766725739694),
    "GBR": (2.20, 8.32935578799489),
    "JPN": (3.25, 1.35105888659274),
    "NLD": (2.95, 9.01362412065194),
    "NOR": (3.43, 68.9331407034111),
    "USA": (4.24, 15.9220948098952),
    "ZAF": (6.57, 8.68304384294431),
}

SEED = 20260726
NBOOT_PANEL = 9999
NBOOT_H1 = 2000
N_PERM_H2 = 20000

def wire_everything() -> None:
    """Wire every master-market input into the H1 engine.

    With ``FOUR_QUADRANT_FROM_DERIVED=1`` and the derived layer present, inputs
    load from ``data/derived/`` and WRDS reconstruction is skipped entirely.
    """
    import os

    if os.environ.get("FOUR_QUADRANT_FROM_DERIVED") == "1":
        from engine import derived_inputs as D

        if not D.available():
            missing = ", ".join(D.missing_files())
            raise FileNotFoundError(
                "FOUR_QUADRANT_FROM_DERIVED=1 was requested, but the derived "
                f"layer is incomplete under data/derived/: {missing}"
            )
        D.wire_from_derived()
        return
    P.wire_all()
    for mkt in ["DEU", "FRA", "ESP", "NLD", "BEL"]:
        cfg = EU.EURO_CONFIG[mkt]
        EU.wire(mkt, cfg, EU.equity_recon(None, mkt, cfg[5]))


def sharpe(x: pd.Series, bill: pd.Series) -> float:
    e = (x - bill).dropna()
    return float(e.mean() / e.std() * np.sqrt(12))


def ann_vol(x: pd.Series, bill: pd.Series | None = None) -> float:
    y = x if bill is None else x - bill
    return float(y.dropna().std() * np.sqrt(12))


def max_drawdown(x: pd.Series) -> float:
    wealth = (1 + x).cumprod().values
    peak = np.maximum.accumulate(wealth)
    return float((wealth / peak - 1).min())


def harder_benchmark(s: dict[str, pd.Series]) -> tuple[str, pd.Series]:
    """Return the benchmark with the higher net excess-return Sharpe."""
    s60 = sharpe(s["b6040"], s["bill"])
    spp = sharpe(s["pp"], s["bill"])
    return ("60/40", s["b6040"]) if s60 >= spp else ("permanent", s["pp"])


def cluster_t(panel: pd.DataFrame, value: str) -> tuple[float, float, float]:
    z = panel[value].to_numpy(float)
    groups = panel["month"].astype(str).to_numpy()
    mean = float(z.mean())
    resid = z - mean
    num = sum(float(resid[groups == g].sum()) ** 2 for g in np.unique(groups))
    se = float(np.sqrt(num) / len(z))
    return mean, se, mean / se


def phi(x: float) -> float:
    return 0.5 * (1 + erf(x / sqrt(2)))


def panel_resampling(panel: pd.DataFrame) -> tuple[float, float]:
    """Few-country inference: a wild country-cluster bootstrap for the t statistic,
    and an exhaustive country sign-flip permutation for the mean.

    With a handful of markets the sign-flip null is small enough to enumerate in
    full, so the permutation p value is exact (the fraction of the ``2**k`` sign
    assignments whose mean is at least the observed one) rather than a Monte Carlo
    estimate. The wild bootstrap for the studentised statistic stays a draw of
    ``NBOOT_PANEL`` Rademacher patterns, seeded for determinism.
    """
    z = panel["z"].to_numpy(float)
    mu, _, t_obs = cluster_t(panel, "z")
    markets = sorted(panel["market"].unique())
    months = sorted(panel["month"].unique())
    mi = {m: i for i, m in enumerate(markets)}
    ti = {t: i for i, t in enumerate(months)}
    mcode = panel["market"].map(mi).to_numpy()
    tcode = panel["month"].map(ti).to_numpy()
    k, tt, n = len(markets), len(months), len(panel)

    country_z = np.bincount(mcode, weights=z, minlength=k)
    resid = z - mu
    cm = np.zeros((k, tt))
    np.add.at(cm, (mcode, tcode), resid)
    country_resid = np.bincount(mcode, weights=resid, minlength=k)
    month_n = np.bincount(tcode, minlength=tt)

    # Exhaustive country sign-flip permutation of the uncentred series: every one of
    # the 2**k sign assignments is evaluated, so the one-sided p value is exact.
    # The all-ones assignment reproduces the observed mean and is a mathematical tie
    # at the threshold; because the permuted mean and ``mu`` are summed in different
    # orders, that tie sits within one unit in the last place of the comparison, so a
    # small tolerance keeps the count stable (the nearest genuine pattern is order
    # 1e-4 away).
    if k > 22:
        raise ValueError(f"exhaustive sign-flip is infeasible for k={k} markets")
    sign_patterns = np.array(list(itertools.product((-1.0, 1.0), repeat=k)))
    perm_means = sign_patterns @ country_z / n
    tie_tolerance = 1e-9
    p_perm = float(np.mean(perm_means >= mu - tie_tolerance))

    # Wild country-cluster bootstrap under H0, studentised by months.
    rng = np.random.default_rng(SEED)
    wild_count = 0
    done = 0
    while done < NBOOT_PANEL:
        b = min(1000, NBOOT_PANEL - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(b, k))
        means = signs @ country_resid / n
        month_sums = signs @ cm - means[:, None] * month_n[None, :]
        ses = np.sqrt(np.sum(month_sums**2, axis=1)) / n
        t_boot = np.divide(means, ses, out=np.zeros_like(means), where=ses > 0)
        wild_count += int(np.sum(t_boot >= t_obs))
        done += b

    return (wild_count + 1) / (NBOOT_PANEL + 1), p_perm


def run_h1() -> tuple[dict[str, dict], pd.DataFrame, dict]:
    print("\n[H1] primary 13-market rerun", flush=True)
    E.run_market.return_series = True
    results: dict[str, dict] = {}
    rows = []
    raw_panel = []
    risk_panel = []
    descriptive = []
    static_rows = []

    for mkt in MARKETS:
        print(f"  {mkt}", flush=True)
        r = E.run_market(
            mkt,
            real_gate=False,
            bootstrap_draws=NBOOT_H1,
        )
        results[mkt] = r
        s = r["series"]
        bench_name, best = harder_benchmark(s)
        ex_s = s["strat"] - s["bill"]
        ex_b = best - s["bill"]
        d = (ex_s - ex_b).dropna()
        z = (
            ex_s / ex_s.rolling(36).std().shift(1)
            - ex_b / ex_b.rolling(36).std().shift(1)
        ).dropna()
        raw_panel.append(
            pd.DataFrame(
                {"d": d.values, "market": mkt, "month": d.index.astype(str)}
            )
        )
        risk_panel.append(
            pd.DataFrame(
                {"z": z.values, "market": mkt, "month": z.index.astype(str)}
            )
        )
        rows.append(
            {
                "market": mkt,
                "country": NAMES[mkt],
                "start": str(s["strat"].index.min()),
                "end": str(s["strat"].index.max()),
                "months": len(s["strat"]),
                "benchmark": bench_name,
                "sharpe_switch": sharpe(s["strat"], s["bill"]),
                "sharpe_6040": sharpe(s["b6040"], s["bill"]),
                "sharpe_permanent": sharpe(s["pp"], s["bill"]),
                "sharpe_matched_static": sharpe(s["matched"], s["bill"]),
                "advantage": sharpe(s["strat"], s["bill"]) - sharpe(best, s["bill"]),
                "advantage_vs_matched": sharpe(s["strat"], s["bill"])
                - sharpe(s["matched"], s["bill"]),
                "p_block_bootstrap": r["p_boot"],
                "ci_low": r["adv_ci_low"],
                "ci_high": r["adv_ci_high"],
                "vol_switch": ann_vol(s["strat"], s["bill"]),
                "vol_6040": ann_vol(s["b6040"], s["bill"]),
                "drawdown_switch": max_drawdown(s["strat"]),
                "drawdown_6040": max_drawdown(s["b6040"]),
                "terminal_wealth_switch": float((1 + s["strat"]).prod()),
                "terminal_wealth_benchmark": float((1 + best).prod()),
                "bond_state_share": r["matched_bond_state_share"],
            }
        )
        for asset in ("eq", "bond", "gold", "bill"):
            series = s[asset]
            descriptive.append(
                {
                    "market": mkt,
                    "asset": asset,
                    "n": int(series.notna().sum()),
                    "mean_annual_pct": float(series.mean() * 1200),
                    "std_annual_pct": float(series.std() * np.sqrt(12) * 100),
                    "min_monthly_pct": float(series.min() * 100),
                    "median_monthly_pct": float(series.median() * 100),
                    "max_monthly_pct": float(series.max() * 100),
                }
            )
        for method in ("b6040", "pp", "matched", "strat"):
            static_rows.append(
                {
                    "market": mkt,
                    "method": method,
                    "sharpe": sharpe(s[method], s["bill"]),
                    "volatility": ann_vol(s[method], s["bill"]),
                    "max_drawdown": max_drawdown(s[method]),
                }
            )

    h1 = pd.DataFrame(rows)
    raw = pd.concat(raw_panel, ignore_index=True)
    risk = pd.concat(risk_panel, ignore_index=True)
    mu_d, se_d, t_d = cluster_t(raw, "d")
    mu_z, se_z, t_z = cluster_t(risk, "z")
    p_wild, p_perm = panel_resampling(risk)

    jack = []
    for omitted in MARKETS:
        sub = risk[risk["market"] != omitted]
        mu, se, t = cluster_t(sub, "z")
        jack.append(
            {
                "omitted_market": omitted,
                "markets": sub["market"].nunique(),
                "observations": len(sub),
                "alpha_monthly": mu,
                "alpha_annualized": mu * np.sqrt(12),
                "se_monthly": se,
                "t": t,
                "p_one_sided": 1 - phi(t),
            }
        )
    jack_df = pd.DataFrame(jack)

    scopes = {
        "all_13": set(MARKETS),
        "external_without_USA": set(MARKETS) - {"USA"},
        "without_five_euro": set(MARKETS) - EURO,
        "one_euro_representative_DEU": (set(MARKETS) - EURO) | {"DEU"},
        "external_without_euro": set(MARKETS) - EURO - {"USA"},
    }
    scope_rows = []
    for label, keep in scopes.items():
        sub = risk[risk["market"].isin(keep)]
        mu, se, t = cluster_t(sub, "z")
        scope_rows.append(
            {
                "sample": label,
                "markets": sub["market"].nunique(),
                "observations": len(sub),
                "month_clusters": sub["month"].nunique(),
                "alpha_annualized": mu * np.sqrt(12),
                "t": t,
                "p_one_sided": 1 - phi(t),
            }
        )

    desc_df = pd.DataFrame(descriptive)
    static_df = pd.DataFrame(static_rows)
    h1.to_csv(OUT / "h1_per_market.csv", index=False)
    raw.to_csv(OUT / "h1_return_panel.csv", index=False)
    risk.to_csv(OUT / "h1_risk_adjusted_panel.csv", index=False)
    jack_df.to_csv(OUT / "h1_jackknife.csv", index=False)
    pd.DataFrame(scope_rows).to_csv(OUT / "h1_scope_robustness.csv", index=False)
    desc_df.to_csv(OUT / "descriptive_asset_returns.csv", index=False)
    static_df.to_csv(OUT / "h1_static_timing.csv", index=False)

    panel_summary = {
        "return_level": {
            "observations": len(raw),
            "month_clusters": raw["month"].nunique(),
            "alpha_monthly": mu_d,
            "alpha_annual_pct": mu_d * 1200,
            "se_monthly": se_d,
            "t": t_d,
            "p_two_sided": 2 * (1 - phi(abs(t_d))),
        },
        "risk_adjusted": {
            "observations": len(risk),
            "month_clusters": risk["month"].nunique(),
            "alpha_monthly": mu_z,
            "alpha_annualized_sharpe": mu_z * np.sqrt(12),
            "se_monthly": se_z,
            "t": t_z,
            "p_one_sided": 1 - phi(t_z),
            "p_wild_country": p_wild,
            "p_country_sign_flip": p_perm,
        },
        "per_market": {
            "positive_advantage": int((h1["advantage"] > 0).sum()),
            "significant_5pct": int((h1["p_block_bootstrap"] < 0.05).sum()),
            "mean_advantage": float(h1["advantage"].mean()),
            "median_advantage": float(h1["advantage"].median()),
            "positive_vs_matched": int((h1["advantage_vs_matched"] > 0).sum()),
            "median_advantage_vs_matched": float(h1["advantage_vs_matched"].median()),
            "mean_volatility_reduction_vs_6040": float(
                (h1["vol_6040"] - h1["vol_switch"]).mean()
            ),
        },
    }
    return results, risk, panel_summary


def energy_keys() -> set[str]:
    smap = pd.read_parquet(RECON / "sector_map.parquet")
    sic = pd.to_numeric(smap["sic"], errors="coerce")
    gsector = smap["gsector"].astype(str)
    flag = (
        (gsector == "10")
        | ((sic >= 1300) & (sic < 1400))
        | ((sic >= 2900) & (sic < 3000))
    )
    return set(smap.loc[flag, "gvkey"].astype(str))


def aggregate_energy(
    data: pd.DataFrame,
    security_col: str,
    firm_col: str,
) -> tuple[pd.Series, pd.DataFrame]:
    rows = []
    for month, x in data.groupby("month", sort=True):
        w = x["w"].to_numpy(float)
        r = x["r"].to_numpy(float)
        total = w.sum()
        if not np.isfinite(total) or total <= 0:
            continue
        wn = w / total
        rows.append(
            {
                "month": month,
                "return": float(np.sum(wn * r)),
                "n_securities": int(x[security_col].nunique()),
                "n_firms": int(x[firm_col].nunique()),
                "hhi": float(np.sum(wn**2)),
                "effective_n": float(1 / np.sum(wn**2)),
            }
        )
    coverage = pd.DataFrame(rows).set_index("month")
    return coverage["return"], coverage


def build_global_energy(mkt: str, ekeys: set[str]) -> tuple[pd.Series, pd.DataFrame]:
    sec = pd.read_parquet(RECON / f"{mkt}_monthend.parquet")
    csho = pd.read_parquet(RECON / f"{mkt}_csho.parquet")
    sec["gvkey"] = sec["gvkey"].astype(str)
    cur = CUR[mkt]
    s = sec[sec["gvkey"].isin(ekeys) & sec["curcdd"].isin(cur)].copy()
    if s.empty:
        raise RuntimeError(f"{mkt}: no listed energy security")
    s["prccd"] = s["prccd"] / s["curcdd"].map(cur)
    s["datadate"] = pd.to_datetime(s["datadate"])
    s["month"] = s["datadate"].dt.to_period("M")
    s = (
        s.sort_values(["gvkey", "iid", "datadate"])
        .drop_duplicates(["gvkey", "iid", "month"], keep="last")
    )
    s["tri"] = s["prccd"] / s["ajexdi"] * s["trfd"]
    c = csho.copy()
    c["gvkey"] = c["gvkey"].astype(str)
    c["datadate"] = pd.to_datetime(c["datadate"])
    c["month"] = c["datadate"].dt.to_period("M")
    c = (
        c.sort_values(["gvkey", "datadate"])
        .drop_duplicates(["gvkey", "month"], keep="last")
    )
    s = s.merge(c[["gvkey", "month", "csho"]], on=["gvkey", "month"], how="left")
    s = s.sort_values(["gvkey", "iid", "month"])
    group = s.groupby(["gvkey", "iid"], sort=False)
    s["csho_ff"] = group["csho"].ffill()
    s["shares"] = s["cshoc"].fillna(s["csho_ff"] * 1e6)
    s = s.dropna(subset=["shares"]).copy()
    s["cap"] = s["prccd"] * s["shares"]
    group = s.groupby(["gvkey", "iid"], sort=False)
    s["r"] = group["tri"].pct_change()
    gap = group["month"].diff().apply(lambda x: getattr(x, "n", None))
    s.loc[gap != 1, "r"] = np.nan
    s["w"] = group["cap"].shift(1)
    s = s.dropna(subset=["r", "w"])
    s = s[(s["r"] > -0.99) & (s["r"] < 10) & (s["w"] > 0)]
    s["security"] = s["gvkey"] + ":" + s["iid"].astype(str)
    return aggregate_energy(s, "security", "gvkey")


def market_energy(mkt: str, ekeys: set[str]) -> tuple[pd.Series, pd.DataFrame]:
    """Return (energy total-return series, coverage) for a market's energy sleeve.

    Offline (``FOUR_QUADRANT_FROM_DERIVED=1``) the aggregated series is read from
    the shipped frozen layer ``data/derived/energy/{market}.parquet``. Otherwise it
    is built from the licensed reconstruction and frozen to that layer, so the
    offline inputs stay in sync with a full run. The frozen file holds only
    market-level aggregates (return, firm/security counts, concentration), never
    firm-level records.
    """
    frozen = DERIVED_ENERGY_DATA / f"{mkt}.parquet"
    if os.environ.get("FOUR_QUADRANT_FROM_DERIVED") == "1":
        coverage = pd.read_parquet(frozen)
        coverage.index = pd.PeriodIndex(coverage["month"], freq="M")
        coverage = coverage.drop(columns="month")
        return coverage["return"], coverage
    energy, coverage = build_usa_energy() if mkt == "USA" else build_global_energy(mkt, ekeys)
    frozen_frame = coverage.copy()
    frozen_frame.index = frozen_frame.index.astype(str)
    frozen_frame.index.name = "month"
    DERIVED_ENERGY_DATA.mkdir(parents=True, exist_ok=True)
    frozen_frame.reset_index().to_parquet(frozen, index=False)
    return energy, coverage


def build_usa_energy() -> tuple[pd.Series, pd.DataFrame]:
    f = RECON / "USA_energy_crsp.parquet"
    if not f.exists():
        raise FileNotFoundError(
            "place USA_energy_crsp.parquet in the configured reconstruction directory"
        )
    s = pd.read_parquet(f)
    s["month"] = pd.to_datetime(s["date"]).dt.to_period("M")
    s = s.sort_values(["permno", "month"])
    group = s.groupby("permno", sort=False)
    s["w"] = group["cap"].shift(1)
    gap = group["month"].diff().apply(lambda x: getattr(x, "n", None))
    s.loc[gap != 1, "ret"] = np.nan
    s = s.rename(columns={"ret": "r"}).dropna(subset=["r", "w"])
    s = s[(s["r"] > -0.99) & (s["r"] < 10) & (s["w"] > 0)]
    s["security"] = s["permno"].astype(str)
    s["firm"] = s["permno"].astype(str)
    return aggregate_energy(s, "security", "firm")


def portfolio(
    returns: pd.DataFrame,
    weight_fn,
    rebalance: str = "M",
    cost: float = 0.004,
    energy_cost: float = 0.005,
) -> pd.Series:
    previous = None
    output = {}
    for month in returns.index:
        target = weight_fn(month)
        if previous is None:
            weights = target
            fee = sum(
                (energy_cost if asset == "energy" else cost) * abs(weight)
                for asset, weight in weights.items()
            )
        else:
            drift = {
                asset: previous[asset] * (1 + returns.loc[previous_month, asset])
                for asset in previous
            }
            total = sum(drift.values())
            drift = {asset: value / total for asset, value in drift.items()}
            no_quarterly_trade = (
                rebalance == "Q"
                and month.month not in (3, 6, 9, 12)
                and weight_fn(month) == weight_fn(previous_month)
            )
            if no_quarterly_trade:
                weights = drift
                fee = 0.0
            else:
                weights = target
                fee = sum(
                    (energy_cost if asset == "energy" else cost)
                    * abs(weights[asset] - drift.get(asset, 0))
                    for asset in weights
                )
        output[month] = (
            sum(weights[asset] * returns.loc[month, asset] for asset in weights) - fee
        )
        previous = weights
        previous_month = month
    return pd.Series(output)


def run_h2_market(mkt: str, energy: pd.Series) -> dict:
    eq = E.equity_tr(mkt)
    y = E.long_yield(mkt)
    bond = construct_monthly_tr(y.to_timestamp().rename("y"))
    bond_r = pd.Series(bond["tr"].values, index=y.index)
    bond_wealth = pd.Series(bond["wealth"].values, index=y.index)
    gold = E.gold_local(mkt)
    gold_r = gold.pct_change()
    bill = E.short_rate(mkt) / 100 / 12
    ratio = (bond_wealth / gold).dropna()
    signal = (ratio >= ratio.rolling(E.WINDOW).mean()).astype(int)
    signal = signal[ratio.rolling(E.WINDOW).count() >= E.WINDOW]
    common = (
        eq.index.intersection(bond_r.index)
        .intersection(gold_r.index)
        .intersection(bill.index)
        .intersection(signal.index)
        .intersection(energy.index)
    )
    returns = pd.DataFrame(
        {
            "eq": eq,
            "bond": bond_r,
            "gold": gold_r,
            "bill": bill,
            "energy": energy,
        }
    ).loc[common]
    lag_signal = signal.shift(1).reindex(common).ffill()

    def state(month) -> int:
        value = lag_signal.loc[month]
        return 1 if np.isnan(value) else int(value)

    rules = {
        "b6040": (
            lambda month: {"eq": 0.60, "bond": 0.40},
            "Q",
        ),
        "energy_static": (
            lambda month: {
                "eq": 0.25,
                "energy": 0.25,
                "bill": 0.25,
                "bond": 0.25,
            },
            "Q",
        ),
        "currency_only": (
            lambda month: {
                "eq": 0.25,
                "bill": 0.25,
                "bond": 0.25 + 0.25 * state(month),
                "gold": 0.25 * (1 - state(month)),
            },
            "M",
        ),
        "currency_energy": (
            lambda month: {
                "eq": 0.25,
                "energy": 0.25,
                "bill": 0.25,
                "bond": 0.25 * state(month),
                "gold": 0.25 * (1 - state(month)),
            },
            "M",
        ),
    }
    series = {
        label: portfolio(returns, fn, rebalance=frequency)
        for label, (fn, frequency) in rules.items()
    }
    return {
        "start": str(common.min()),
        "end": str(common.max()),
        "months": len(common),
        "returns": returns,
        "series": series,
        **{f"sharpe_{key}": sharpe(value, returns["bill"]) for key, value in series.items()},
    }


def exposure_table() -> pd.DataFrame:
    rows = [
        {
            "market": market,
            "energy_intensity_mj_per_2021_ppp_gdp": values[0],
            "fuel_exports_pct_merchandise_exports": values[1],
        }
        for market, values in EXPOSURE_2021.items()
    ]
    data = pd.DataFrame(rows)
    data["rank_energy_intensity"] = data[
        "energy_intensity_mj_per_2021_ppp_gdp"
    ].rank(method="average")
    data["rank_fuel_exports"] = data[
        "fuel_exports_pct_merchandise_exports"
    ].rank(method="average")
    data["exposure_mean_rank"] = data[
        ["rank_energy_intensity", "rank_fuel_exports"]
    ].mean(axis=1)
    return data.sort_values("exposure_mean_rank")


def run_h2() -> tuple[pd.DataFrame, dict]:
    print("\n[H2] complete 13-market energy rerun", flush=True)
    keys = set() if os.environ.get("FOUR_QUADRANT_FROM_DERIVED") == "1" else energy_keys()
    exposures = exposure_table()
    exposures.to_csv(OUT / "h2_world_bank_exposure_2021.csv", index=False)

    rows = []
    coverage_rows = []
    for mkt in MARKETS:
        print(f"  {mkt}", flush=True)
        energy, coverage = market_energy(mkt, keys)
        result = run_h2_market(mkt, energy)
        in_window = coverage.reindex(result["returns"].index).dropna()
        coverage_rows.append(
            {
                "market": mkt,
                "raw_start": str(coverage.index.min()),
                "raw_end": str(coverage.index.max()),
                "raw_months": len(coverage),
                "evaluation_start": result["start"],
                "evaluation_end": result["end"],
                "evaluation_months": result["months"],
                "minimum_firms": int(in_window["n_firms"].min()),
                "median_firms": float(in_window["n_firms"].median()),
                "maximum_firms": int(in_window["n_firms"].max()),
                "median_effective_n": float(in_window["effective_n"].median()),
                "minimum_effective_n": float(in_window["effective_n"].min()),
                "median_hhi": float(in_window["hhi"].median()),
                "months_below_5_firms": int((in_window["n_firms"] < 5).sum()),
                "share_months_below_5_firms": float((in_window["n_firms"] < 5).mean()),
            }
        )
        rows.append(
            {
                "market": mkt,
                "country": NAMES[mkt],
                "start": result["start"],
                "end": result["end"],
                "months": result["months"],
                "sharpe_6040": result["sharpe_b6040"],
                "sharpe_energy_static": result["sharpe_energy_static"],
                "sharpe_currency_only": result["sharpe_currency_only"],
                "sharpe_currency_energy": result["sharpe_currency_energy"],
                "augmentation_advantage": result["sharpe_currency_energy"]
                - result["sharpe_currency_only"],
            }
        )

    h2 = pd.DataFrame(rows).merge(exposures, on="market", how="left")
    coverage_df = pd.DataFrame(coverage_rows)
    h2.to_csv(OUT / "h2_per_market.csv", index=False)
    coverage_df.to_csv(OUT / "h2_energy_coverage.csv", index=False)

    x = h2["exposure_mean_rank"].to_numpy(float)
    y = h2["augmentation_advantage"].to_numpy(float)
    X = sm.add_constant(x)
    fit = sm.OLS(y, X).fit()
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(SEED)
    permuted = np.empty(N_PERM_H2)
    for b in range(N_PERM_H2):
        permuted[b] = spearmanr(rng.permutation(x), y).statistic
    p_perm = float((np.sum(permuted >= rho) + 1) / (N_PERM_H2 + 1))
    fitted = fit.predict(X)
    harms = int((y < 0).sum())
    sign_p = float(binomtest(harms, n=len(y), p=0.5, alternative="greater").pvalue)

    summary = {
        "markets": len(h2),
        "augmentation_negative": harms,
        "augmentation_mean": float(y.mean()),
        "augmentation_median": float(np.median(y)),
        "sign_test_p_one_sided": sign_p,
        "ols_intercept": float(fit.params[0]),
        "ols_intercept_t": float(fit.tvalues[0]),
        "ols_slope": float(fit.params[1]),
        "ols_slope_se": float(fit.bse[1]),
        "ols_slope_t": float(fit.tvalues[1]),
        "ols_slope_p_one_sided_positive": float(1 - phi(float(fit.tvalues[1]))),
        "spearman_rho": rho,
        "spearman_permutation_p_one_sided": p_perm,
        "fitted_values_negative": int((fitted < 0).sum()),
        "fitted_min": float(fitted.min()),
        "fitted_max": float(fitted.max()),
        "markets_median_effective_n_below_5": coverage_df.loc[
            coverage_df["median_effective_n"] < 5, "market"
        ].tolist(),
    }
    return h2, summary


def pooled_h3_inference(e_amp, e_bin, months) -> list[dict]:
    """Pooled H3 Sharpe-difference inference over the market-month panel.

    Returns one row per method: the calendar-month cluster (a zero-lag HAC), the
    twelve-lag calendar-month HAC that the paper reports, and a circular block
    bootstrap over calendar months. All three sum the moment conditions by calendar
    month so the cross-market dependence is respected.
    """
    cluster = pooled_calendar_hac(e_amp, e_bin, months, lags=0)
    hac12 = pooled_calendar_hac(e_amp, e_bin, months, lags=12)
    boot = pooled_calendar_block_bootstrap(
        e_amp, e_bin, months, block=12, draws=4999, seed=SEED
    )
    rows = []
    for method, lags, stat in (
        ("calendar_month_cluster", 0, cluster),
        ("calendar_month_hac12", 12, hac12),
    ):
        d, se, z, p_amp, p_bin, p_two, n, n_months = stat
        rows.append(
            {
                "method": method,
                "observations": n,
                "calendar_months": n_months,
                "markets": len(MARKETS),
                "lags": lags,
                "block_length": "",
                "bootstrap_draws": 0,
                "difference_amplitude_minus_binary": d,
                "standard_error": se,
                "z_statistic": z,
                "ci95_low": d - 1.959963984540054 * se,
                "ci95_high": d + 1.959963984540054 * se,
                "p_amplitude_greater": p_amp,
                "p_binary_greater": p_bin,
                "p_two_sided": p_two,
            }
        )
    bd, bse, bcl, bch, bp_amp, bp_bin, bp_two, bn, bn_months = boot
    rows.append(
        {
            "method": "calendar_month_block_bootstrap_12",
            "observations": bn,
            "calendar_months": bn_months,
            "markets": len(MARKETS),
            "lags": "",
            "block_length": 12,
            "bootstrap_draws": 4999,
            "difference_amplitude_minus_binary": bd,
            "standard_error": bse,
            "z_statistic": "",
            "ci95_low": bcl,
            "ci95_high": bch,
            "p_amplitude_greater": bp_amp,
            "p_binary_greater": bp_bin,
            "p_two_sided": bp_two,
        }
    )
    return rows


def run_h3() -> tuple[pd.DataFrame, dict]:
    print("\n[H3] binary versus amplitude, 13 markets", flush=True)
    rows = []
    pooled_amp = []
    pooled_bin = []
    pooled_months = []
    for mkt in MARKETS:
        print(f"  {mkt}", flush=True)
        returns, binary_signal, amplitude_fraction = series_for(mkt)
        binary = run_variant(returns, binary_signal)
        amplitude = run_variant(returns, amplitude_fraction)
        diff, p_value, n = lw_test(amplitude, binary, returns["bill"])
        binary_excess = (binary - returns["bill"]).dropna()
        amplitude_excess = (amplitude - returns["bill"]).reindex(binary_excess.index)
        rows.append(
            {
                "market": mkt,
                "country": NAMES[mkt],
                "start": str(binary.index.min()),
                "end": str(binary.index.max()),
                "months": n,
                "sharpe_binary": float(
                    binary_excess.mean() / binary_excess.std() * np.sqrt(12)
                ),
                "sharpe_amplitude": float(
                    amplitude_excess.mean() / amplitude_excess.std() * np.sqrt(12)
                ),
                "difference_amplitude_minus_binary": diff,
                "p_amplitude_greater": p_value,
                "terminal_wealth_ratio_amplitude_binary": float(
                    (1 + amplitude).prod() / (1 + binary).prod()
                ),
            }
        )
        aligned = pd.DataFrame(
            {"amp": amplitude_excess, "bin": binary_excess}
        ).dropna()
        pooled_amp.append(aligned["amp"].to_numpy(float))
        pooled_bin.append(aligned["bin"].to_numpy(float))
        pooled_months.append(aligned.index.astype(str).to_numpy())

    h3 = pd.DataFrame(rows)
    h3.to_csv(OUT / "h3_per_market.csv", index=False)

    # Pooled inference respects the panel: moment conditions are summed by calendar
    # month before the HAC, rather than stacking the markets into one long series.
    e_amp = np.concatenate(pooled_amp)
    e_bin = np.concatenate(pooled_bin)
    e_months = np.concatenate(pooled_months)
    pooled_rows = pooled_h3_inference(e_amp, e_bin, e_months)
    pd.DataFrame(pooled_rows).to_csv(
        OUT / "h3_pooled_calendar_inference.csv", index=False
    )
    hac12 = next(r for r in pooled_rows if r["method"] == "calendar_month_hac12")

    summary = {
        "markets": len(h3),
        "amplitude_beats_binary": int(
            (h3["difference_amplitude_minus_binary"] > 0).sum()
        ),
        "amplitude_significant_5pct": int(
            (
                (h3["difference_amplitude_minus_binary"] > 0)
                & (h3["p_amplitude_greater"] < 0.05)
            ).sum()
        ),
        "pooled_observations": hac12["observations"],
        "pooled_calendar_months": hac12["calendar_months"],
        "pooled_difference": hac12["difference_amplitude_minus_binary"],
        "pooled_standard_error": hac12["standard_error"],
        "pooled_p_amplitude_greater": hac12["p_amplitude_greater"],
        "pooled_p_binary_greater": hac12["p_binary_greater"],
        "terminal_wealth_ratio_below_one": int(
            (h3["terminal_wealth_ratio_amplitude_binary"] < 1).sum()
        ),
    }
    return h3, summary


def run_inflation_validity(h1_results: dict[str, dict]) -> dict:
    print("\n[mechanism] inflation-state correspondence", flush=True)
    rows = []
    panels = []
    for mkt in MARKETS:
        signal = h1_results[mkt]["series"]["signal"]
        inflation = E.cpi_index(mkt).pct_change(12) * 100
        data = pd.concat(
            [inflation.rename("inflation"), signal.rename("signal")], axis=1
        ).dropna()
        X = sm.add_constant(data["signal"].astype(float))
        fit = sm.OLS(data["inflation"], X).fit(
            cov_type="HAC", cov_kwds={"maxlags": 12}
        )
        rows.append(
            {
                "market": mkt,
                "observations": len(data),
                "coefficient_bond_state": float(fit.params["signal"]),
                "se_hac12": float(fit.bse["signal"]),
                "t_hac12": float(fit.tvalues["signal"]),
                "p_two_sided": float(fit.pvalues["signal"]),
            }
        )
        panels.append(data.assign(market=mkt, month=data.index.astype(str)))
    individual = pd.DataFrame(rows)
    panel = pd.concat(panels, ignore_index=True)
    dummies = pd.get_dummies(panel["market"], drop_first=True, dtype=float)
    X = pd.concat(
        [
            panel[["signal"]].astype(float).reset_index(drop=True),
            dummies.reset_index(drop=True),
        ],
        axis=1,
    )
    X = sm.add_constant(X)
    fit = sm.OLS(panel["inflation"].to_numpy(), X).fit(
        cov_type="cluster", cov_kwds={"groups": panel["month"].to_numpy()}
    )
    individual.to_csv(OUT / "inflation_signal_by_market.csv", index=False)
    return {
        "markets_negative": int((individual["coefficient_bond_state"] < 0).sum()),
        "pooled_observations": len(panel),
        "pooled_beta": float(fit.params["signal"]),
        "pooled_se": float(fit.bse["signal"]),
        "pooled_t": float(fit.tvalues["signal"]),
        "pooled_p_two_sided": float(fit.pvalues["signal"]),
    }


def run_rdd(risk_panel: pd.DataFrame) -> pd.DataFrame:
    print("\n[robustness] 1999 break", flush=True)
    data = risk_panel.copy()
    periods = pd.PeriodIndex(data["month"], freq="M")
    data["running_month"] = (periods.year - 1999) * 12 + (periods.month - 1)
    data["post"] = (data["running_month"] >= 0).astype(float)
    data["euro"] = data["market"].isin(EURO).astype(float)
    rows = []
    for bandwidth in (48, 60, 84, 120):
        for diff in (False, True):
            d = data[data["running_month"].abs() <= bandwidth].copy()
            X = pd.DataFrame(
                {
                    "post": d["post"],
                    "running": d["running_month"],
                    "running_post": d["running_month"] * d["post"],
                }
            )
            market_dummies = pd.get_dummies(d["market"], drop_first=True, dtype=float)
            X = pd.concat([X.reset_index(drop=True), market_dummies.reset_index(drop=True)], axis=1)
            if diff:
                X["post_euro"] = d["post"].to_numpy() * d["euro"].to_numpy()
                X["euro"] = d["euro"].to_numpy()
            X = sm.add_constant(X)
            fit = sm.OLS(d["z"].to_numpy(), X.to_numpy()).fit(
                cov_type="cluster", cov_kwds={"groups": d["month"].to_numpy()}
            )
            names = list(X.columns)
            post_i = names.index("post")
            row = {
                "bandwidth_months": bandwidth,
                "difference_in_discontinuities": diff,
                "observations": len(d),
                "break_monthly": float(fit.params[post_i]),
                "break_annualized": float(fit.params[post_i] * np.sqrt(12)),
                "break_t": float(fit.tvalues[post_i]),
                "break_p_two_sided": float(fit.pvalues[post_i]),
            }
            if diff:
                euro_i = names.index("post_euro")
                row.update(
                    {
                        "additional_euro_break_monthly": float(fit.params[euro_i]),
                        "additional_euro_break_t": float(fit.tvalues[euro_i]),
                        "additional_euro_break_p_two_sided": float(
                            fit.pvalues[euro_i]
                        ),
                    }
                )
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "rdd_1999.csv", index=False)
    return result


def run_sensitivity() -> dict:
    print("\n[robustness] costs and signal window", flush=True)
    E.run_market.return_series = False
    cost_rows = []
    for cost in (0.0, 0.002, 0.004, 0.006, 0.008, 0.010):
        print(f"  cost={cost:.3f}", flush=True)
        values = [
            E.run_market(
                mkt,
                real_gate=False,
                cost_oneway=cost,
                bootstrap_draws=0,
            )["advantage"]
            for mkt in MARKETS
        ]
        cost_rows.append(
            {
                "cost_oneway": cost,
                "median_advantage": float(np.median(values)),
                "mean_advantage": float(np.mean(values)),
                "positive_markets": int(np.sum(np.asarray(values) > 0)),
            }
        )
    window_rows = []
    for window in (60, 72, 84, 96, 108, 120):
        print(f"  window={window}", flush=True)
        values = [
            E.run_market(
                mkt,
                real_gate=False,
                window=window,
                bootstrap_draws=0,
            )["advantage"]
            for mkt in MARKETS
        ]
        window_rows.append(
            {
                "window_months": window,
                "median_advantage": float(np.median(values)),
                "mean_advantage": float(np.mean(values)),
                "positive_markets": int(np.sum(np.asarray(values) > 0)),
            }
        )
    costs = pd.DataFrame(cost_rows)
    windows = pd.DataFrame(window_rows)
    costs.to_csv(OUT / "h1_cost_sensitivity.csv", index=False)
    windows.to_csv(OUT / "h1_window_sensitivity.csv", index=False)
    return {"costs": cost_rows, "windows": window_rows}


def render_h1_forest() -> None:
    """Render the H1 forest figure with the reference-reselection intervals.

    Inputs: ``h1_per_market.csv`` (advantage and raw intervals) and
    ``h1_reference_reselection_bootstrap.csv`` (intervals that reselect
    max(60/40, permanent) in every bootstrap draw), both from ``OUT``. The
    reselection audit is produced by ``analysis.supplementary_inference``, so
    this step must run after it; the file is therefore always present.
    Output: ``h1_advantage_13_markets.png`` (language suffix via ``figpath``).
    Purpose: render the forest once, post-reselection, so the French and English
    figures are identical and stable across reruns.
    """
    print("\n[figures] H1 forest (post-reselection)", flush=True)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#444444",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    # H1 forest.  The dated audit file reselects max(60/40, permanent) in
    # every bootstrap draw.  Those intervals are always preferred.
    h1 = pd.read_csv(OUT / "h1_per_market.csv")
    reselected = pd.read_csv(OUT / "h1_reference_reselection_bootstrap.csv")[
        ["market", "bootstrap_ci95_low", "bootstrap_ci95_high"]
    ].rename(
        columns={
            "bootstrap_ci95_low": "ci_low_reselected",
            "bootstrap_ci95_high": "ci_high_reselected",
        }
    )
    h1 = h1.merge(reselected, on="market", how="left")
    h1["ci_low"] = h1["ci_low_reselected"].fillna(h1["ci_low"])
    h1["ci_high"] = h1["ci_high_reselected"].fillna(h1["ci_high"])

    # H1 forest
    d = h1.sort_values("advantage").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 7.1))
    y = np.arange(len(d))
    for yi, row in zip(y, d.itertuples()):
        colour = BLUE if row.advantage >= 0 else RED
        ax.plot([row.ci_low, row.ci_high], [yi, yi], color=colour, lw=1.4)
        ax.scatter(row.advantage, yi, s=48, color=colour, zorder=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([market_name(m, NAMES[m]) for m in d["market"]])
    ax.set_xlabel(
        L(
            r"$\mathrm{SR}_{\mathrm{bascule}}"
            r"-\max(\mathrm{SR}_{60/40},\mathrm{SR}_{\mathrm{permanent}})$",
            r"$\mathrm{SR}_{\mathrm{switch}}"
            r"-\max(\mathrm{SR}_{60/40},\mathrm{SR}_{\mathrm{permanent}})$",
        )
    )
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(
        figpath(OUT / "h1_advantage_13_markets.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def make_plots(h2: pd.DataFrame, h3: pd.DataFrame) -> None:
    print("\n[figures] separate audit outputs", flush=True)
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 10.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#444444",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    # H2 four-arm decomposition, all 13 countries.
    d = h2.sort_values("augmentation_advantage").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.2, 7.3))
    y = np.arange(len(d))[::-1]
    labels = {
        "sharpe_6040": L(r"$60/40$ local", r"local $60/40$"),
        "sharpe_energy_static": L(r"énergie statique", r"static energy"),
        "sharpe_currency_only": L(r"bascule monétaire", r"monetary switch"),
        "sharpe_currency_energy": L(
            r"bascule monétaire $+$ énergie",
            r"monetary switch $+$ energy",
        ),
    }
    colours = {
        "sharpe_6040": GREY,
        "sharpe_energy_static": GOLD,
        "sharpe_currency_only": BLUE,
        "sharpe_currency_energy": ORANGE,
    }
    for position, row in zip(y, d.itertuples()):
        values = [getattr(row, key) for key in labels]
        ax.plot(
            [min(values), max(values)],
            [position, position],
            color=GRID,
            lw=2.2,
            zorder=1,
        )
        for key in labels:
            ax.scatter(
                getattr(row, key),
                position,
                s=56,
                color=colours[key],
                zorder=3,
                label=labels[key] if position == y[0] else None,
            )
    ax.set_yticks(y)
    ax.set_yticklabels([market_name(m, NAMES[m]) for m in d["market"]])
    ax.set_xlabel(L(r"ratio de Sharpe net", r"net Sharpe ratio"))
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(
        figpath(OUT / "h2_decomposition_13_markets.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # H2 exposure gradient.
    x = h2["exposure_mean_rank"].to_numpy(float)
    yv = h2["augmentation_advantage"].to_numpy(float)
    slope, intercept = np.polyfit(x, yv, 1)
    xs = np.linspace(x.min() - 0.3, x.max() + 0.3, 100)
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    ax.axhline(0, color=INK, lw=0.8)
    ax.plot(
        xs,
        intercept + slope * xs,
        color=GREY,
        lw=1.3,
        ls="--",
        label=L(
            rf"$\widehat{{\beta}}={slope:+.4f}$ par rang",
            rf"$\widehat{{\beta}}={slope:+.4f}$ per rank",
        ),
    )
    ax.scatter(x, yv, color=BLUE, s=58, zorder=3)
    for row in h2.itertuples():
        ax.annotate(
            row.market,
            (row.exposure_mean_rank, row.augmentation_advantage),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.5,
        )
    ax.set_xlabel(
        L(
            r"$E_i$ : rang moyen d'exposition énergétique en 2021",
            r"$E_i$: mean energy-exposure rank in 2021",
        )
    )
    ax.set_ylabel(
        L(
            r"$A_i^{\mathrm{aug}}"
            r"=\mathrm{SR}_{\mathrm{monetaire+energie}}"
            r"-\mathrm{SR}_{\mathrm{monetaire}}$",
            r"$A_i^{\mathrm{aug}}"
            r"=\mathrm{SR}_{\mathrm{monetary+energy}}"
            r"-\mathrm{SR}_{\mathrm{monetary}}$",
        )
    )
    ax.legend(frameon=False, loc="best")
    ax.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figpath(OUT / "h2_exposure_gradient_13_markets.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # H3 dumbbell.
    d = h3.sort_values("difference_amplitude_minus_binary").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    y = np.arange(len(d))[::-1]
    for position, row in zip(y, d.itertuples()):
        ax.plot(
            [row.sharpe_binary, row.sharpe_amplitude],
            [position, position],
            color=GRID,
            lw=2.4,
            zorder=1,
        )
        ax.scatter(
            row.sharpe_binary,
            position,
            s=58,
            color=BLUE,
            label=(L("binaire", "binary") if position == y[0] else None),
            zorder=3,
        )
        ax.scatter(
            row.sharpe_amplitude,
            position,
            s=58,
            color=ORANGE,
            label=(L("graduée", "graded") if position == y[0] else None),
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([market_name(m, NAMES[m]) for m in d["market"]])
    ax.set_xlabel(L(r"ratio de Sharpe net", r"net Sharpe ratio"))
    ax.legend(frameon=False)
    ax.xaxis.grid(True, color=GRID, lw=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(
        figpath(OUT / "h3_binary_amplitude_13_markets.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(
        {
            "market": MARKETS,
            "country": [NAMES[m] for m in MARKETS],
            "included_h1": True,
            "included_h2": True,
            "included_h3": True,
            "sample_rule": "master universe; maximum valid history per market",
        }
    )
    manifest.to_csv(OUT / "sample_manifest.csv", index=False)

    wire_everything()
    h1_results, risk_panel, h1_summary = run_h1()
    h2, h2_summary = run_h2()
    h3, h3_summary = run_h3()
    inflation_summary = run_inflation_validity(h1_results)
    rdd = run_rdd(risk_panel)
    sensitivity = run_sensitivity()
    # The H1 forest figure is rendered by a separate step
    # (``figures.figure_h1_forest``) that runs after the reference-reselection
    # audit in ``analysis.supplementary_inference``, so it always uses the
    # reselected intervals in both language builds. The remaining figures do not
    # depend on any later step and stay here.
    make_plots(h2, h3)

    summary = {
        "run_date": "2026-07-26",
        "master_markets": MARKETS,
        "sample_policy": {
            "country_universe": "same 13 markets in H1, H2, and H3",
            "time_window": "maximum valid history per market; no balanced-calendar truncation",
            "cpi_gate": False,
            "h2_energy_minimum": "one valid listed energy firm; concentration reported, never excluded",
            "h2_exposure_reference_year": 2021,
            "h2_exposure_source": "World Bank API snapshot checked 2026-07-26",
            "paper_modified": False,
        },
        "h1": h1_summary,
        "h2": h2_summary,
        "h3": h3_summary,
        "inflation_validity": inflation_summary,
        "rdd_1999": rdd.to_dict(orient="records"),
        "sensitivity": sensitivity,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\ncomplete rerun written to {OUT.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
