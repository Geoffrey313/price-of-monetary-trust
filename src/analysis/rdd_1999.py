"""Estimate the descriptive discontinuity around January 1999.

Inputs: in-memory H1 return series for euro and comparison markets.
Outputs: printed local-window regressions; ``analysis.run_full_sample`` writes
the corresponding published RDD CSV.
Purpose: describe, without a causal claim, the timing of the H1 advantage.

Exploratory panel regression-discontinuity-in-time at the 1999-01 cutoff.
Outcome: monthly risk-adjusted differential z_it = strat_ex/sd36lag - bestbench_ex/sd36lag.
Running variable: months relative to 1999-01. Local-linear on each side within bandwidth h, country
FE, SEs clustered by calendar month. Reports (a) the pooled discontinuity (the 1999 break), (b) a
difference-in-discontinuities euro vs non-euro (the best available attribution attempt).
Caveat (logged): RD-in-time identifies the BREAK, not its cause — euro/Great-Moderation/inflation-
targeting are collinear at 1999 and cannot be separated. Run: python rdd_1999.py
"""
import numpy as np, pandas as pd
import statsmodels.api as sm
from analysis import euro_experiment as EU
from engine import market_wiring as P

GROUP = {"DEU": "euro", "FRA": "euro", "ESP": "euro", "NLD": "euro", "BEL": "euro",
         "CHE": "eurofloat", "GBR": "eurofloat", "NOR": "eurofloat",
         "USA": "noneuro", "CAN": "noneuro", "AUS": "noneuro", "JPN": "noneuro", "ZAF": "noneuro"}


def build_panel():
    P.wire_all()
    # wire all 5 euro members via the same euro-legacy machinery (overrides wire_all's DE/FR)
    for mkt in ["DEU", "FRA", "ESP", "NLD", "BEL"]:
        cfg = EU.EURO_CONFIG[mkt]
        EU.wire(mkt, cfg, EU.equity_recon(None, mkt, cfg[5]))
    from engine import engine as E
    E.run_market.return_series = True
    rows = []
    for m in GROUP:
        s = E.run_market(m)["series"]
        best = s["b6040"] if (s["b6040"] - s["bill"]).mean() >= (s["pp"] - s["bill"]).mean() else s["pp"]
        ex_s = (s["strat"] - s["bill"]); ex_b = (best - s["bill"])
        z = (ex_s / ex_s.rolling(36).std().shift(1) - ex_b / ex_b.rolling(36).std().shift(1)).dropna()
        df = pd.DataFrame({"z": z})
        df["r"] = [(p.year - 1999) * 12 + (p.month - 1) for p in z.index]   # months since 1999-01
        df["D"] = (df["r"] >= 0).astype(float)
        df["mkt"] = m; df["grp"] = GROUP[m]
        df["month"] = z.index.astype(str)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def rdd(panel, h, diff=False):
    d = panel[panel["r"].abs() <= h].copy()
    X = pd.DataFrame({"D": d["D"], "r": d["r"], "rD": d["r"] * d["D"]})
    for m in sorted(d["mkt"].unique())[1:]:
        X[f"fe_{m}"] = (d["mkt"] == m).astype(float)
    X["const"] = 1.0
    if diff:
        euro = (d["grp"] == "euro").astype(float).values
        X["D_euro"] = d["D"].values * euro
        X["euro"] = euro
    res = sm.OLS(d["z"].values, X.values).fit(cov_type="cluster",
                                              cov_kwds={"groups": d["month"].values})
    names = list(X.columns)
    b = res.params[names.index("D")]; t = res.tvalues[names.index("D")]; p = res.pvalues[names.index("D")]
    out = f"  h=±{h:>3d}m (n={len(d)}): discontinuité à 1999 = {b:+.4f}  t={t:+.2f}  p={p:.3f}"
    if diff:
        bd = res.params[names.index("D_euro")]; td = res.tvalues[names.index("D_euro")]
        pd_ = res.pvalues[names.index("D_euro")]
        out += f"   |   surcroît EURO (diff-in-disc) = {bd:+.4f}  t={td:+.2f}  p={pd_:.3f}"
    return out


def main():
    panel = build_panel()
    print(f"panel: {len(panel)} mois-marché, {panel['mkt'].nunique()} marchés spanning 1999")
    print("\n(a) RUPTURE À 1999 — discontinuité du différentiel risque-ajusté (tous marchés):")
    for h in (48, 60, 84, 120):
        print(rdd(panel, h))
    print("\n(b) DIFFÉRENCE-DE-DISCONTINUITÉS — surcroît de saut spécifique à l'euro:")
    for h in (60, 84, 120):
        print(rdd(panel, h, diff=True))
    print("\n  (rappel: 'surcroît euro' isole le saut euro du saut commun; s'il est ~0/n.s.,"
          "\n   la rupture de 1999 n'est pas attribuable à l'euro au-delà de l'effet commun)")


if __name__ == "__main__":
    main()
