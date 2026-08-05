"""Referee check M10, part 1: predictive inflation-state regression.

Inputs: the wired thirteen-market engine (signal series and CPI levels from
the derived layer).
Outputs: ``results/full-sample/mechanism_predictive.csv``.
Purpose: strengthen the mechanism check.  The published check regresses the
CONTEMPORANEOUS twelve-month inflation rate on the binary bond state and is
fragile under Driscoll and Kraay covariance.  This module runs the same
regression with FUTURE inflation over months t+1..t+12 as the dependent
variable (the same twelve-month CPI percentage change, led twelve months so
the window starts strictly after the signal month t), and reports both
specifications under the identical inference menu.

Mirroring rules.  Data construction, sample rules, and covariance
implementations replicate the published check exactly:

* inflation is ``E.cpi_index(market).pct_change(12) * 100`` as in
  ``run_full_sample.run_inflation_validity`` and in the published joint
  inference (``inflation_signal_joint_inference.csv``);
* the state is the engine's applied (lagged) signal series
  ``run_market(...)['series']['signal']`` with ``real_gate=False``, exactly
  the series the published check uses;
* the estimator is ``linearmodels.panel.PanelOLS`` with entity effects, fit
  with ``debiased=True`` under (a) calendar-month clustering and (b) the
  Driscoll and Kraay Bartlett kernel at 12 and 24 lags, with normal p-values,
  byte-for-byte the covariance code path of the published joint inference.

The contemporaneous specification is re-run through this harness and
reconciled against the published CSV before anything is written; the module
aborts if the mirror is not exact.
"""
from __future__ import annotations

import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from analysis import run_full_sample as R
from common.names import MARKETS
from common.paths import FULL_SAMPLE_RESULTS
from engine import engine as E

OUT = FULL_SAMPLE_RESULTS
PUBLISHED_FILE = OUT / "inflation_signal_joint_inference.csv"
OUT_FILE = OUT / "mechanism_predictive.csv"

COVARIANCE_SPECS = [
    ("calendar_month_cluster", {"cov_type": "clustered", "cluster_time": True}),
    (
        "driscoll_kraay_12",
        {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 12},
    ),
    (
        "driscoll_kraay_24",
        {"cov_type": "kernel", "kernel": "bartlett", "bandwidth": 24},
    ),
]

RECONCILE_TOLERANCE = 1e-8


def build_panel(lead_months: int) -> pd.DataFrame:
    """Stack (inflation, signal) market-months, leading inflation if asked.

    ``lead_months=0`` reproduces the published contemporaneous panel.
    ``lead_months=12`` regresses the twelve-month inflation rate realised
    over t+1..t+12 on the state at t: because the inflation variable at
    month m is the CPI change over m-11..m, shifting it back twelve months
    aligns the window t+1..t+12 with the signal month t.
    """
    panels: list[pd.DataFrame] = []
    for market in MARKETS:
        result = E.run_market(market, real_gate=False, bootstrap_draws=0)
        signal = result["series"]["signal"]
        inflation = E.cpi_index(market).pct_change(12) * 100
        if lead_months:
            inflation = inflation.shift(-lead_months)
        data = pd.concat(
            [inflation.rename("inflation"), signal.rename("signal")], axis=1
        ).dropna()
        panels.append(data.assign(market=market, month=data.index.astype(str)))
    return pd.concat(panels, ignore_index=True)


def fit_panel(panel: pd.DataFrame) -> dict[str, float]:
    """Entity-effects pooled fit under the published covariance menu."""
    panel = panel.copy()
    panel["time"] = pd.PeriodIndex(panel["month"], freq="M").to_timestamp()
    indexed = panel.set_index(["market", "time"]).sort_index()
    model = PanelOLS(
        indexed["inflation"].astype(float),
        indexed[["signal"]].astype(float),
        entity_effects=True,
    )
    out: dict[str, float] = {"n_obs": float(len(panel))}
    for name, keywords in COVARIANCE_SPECS:
        fit = model.fit(debiased=True, **keywords)
        coefficient = float(fit.params["signal"])
        standard_error = float(fit.std_errors["signal"])
        statistic = coefficient / standard_error
        p_two_sided = float(2 * norm.sf(abs(statistic)))
        if name == "calendar_month_cluster":
            out.update(
                beta=coefficient,
                se=standard_error,
                t=statistic,
                p_clustered=p_two_sided,
            )
        elif name == "driscoll_kraay_12":
            out["p_dk12"] = p_two_sided
        else:
            out["p_dk24"] = p_two_sided
    return out


def reconcile_contemporaneous(fitted: dict[str, float]) -> None:
    """Abort unless the harness reproduces the published numbers exactly."""
    published = pd.read_csv(PUBLISHED_FILE).set_index("method")
    checks = [
        ("beta", "calendar_month_cluster", "coefficient_bond_state_pp"),
        ("se", "calendar_month_cluster", "standard_error"),
        ("t", "calendar_month_cluster", "t_statistic"),
        ("p_clustered", "calendar_month_cluster", "p_two_sided"),
        ("p_dk12", "driscoll_kraay_12", "p_two_sided"),
        ("p_dk24", "driscoll_kraay_24", "p_two_sided"),
        ("n_obs", "calendar_month_cluster", "observations"),
    ]
    for key, method, column in checks:
        gap = abs(fitted[key] - float(published.loc[method, column]))
        if gap > RECONCILE_TOLERANCE:
            raise RuntimeError(
                "Contemporaneous mirror is NOT exact: "
                f"{key} differs from the published {method}/{column} "
                f"by {gap:.3e}"
            )
    print(
        "  reconciliation passed: contemporaneous run matches the published "
        f"joint inference to {RECONCILE_TOLERANCE:g}",
        flush=True,
    )


def main() -> int:
    R.wire_everything()
    E.run_market.return_series = True

    print("[M10] contemporaneous mirror", flush=True)
    contemporaneous = fit_panel(build_panel(lead_months=0))
    reconcile_contemporaneous(contemporaneous)

    print("[M10] predictive regression, inflation over t+1..t+12", flush=True)
    predictive = fit_panel(build_panel(lead_months=12))

    columns = ["beta", "se", "t", "p_clustered", "p_dk12", "p_dk24", "n_obs"]
    table = pd.DataFrame(
        [
            {"specification": "contemporaneous", **contemporaneous},
            {"specification": "predictive_t+1..t+12", **predictive},
        ]
    )[["specification", *columns]]
    table["n_obs"] = table["n_obs"].astype(int)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_FILE, index=False)
    print(table.to_string(index=False), flush=True)
    print(f"wrote {OUT_FILE}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
